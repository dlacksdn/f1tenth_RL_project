"""F1Tenth env adapter for dreamerv3-torch (Phase 2-0, decision #22/#25).

Bridges our gymnasium-API ``F110GymnasiumWrapper`` to the *internal* dreamerv3
convention used by ``envs/dmc.py`` and the ``envs/wrappers.py`` chain:

  * old-gym 4-tuple ``step -> (obs, reward, done, info)`` (not gymnasium 5-tuple),
  * ``reset() -> obs`` (no seed/options args, no info),
  * obs dict carries ``is_first`` / ``is_terminal`` at runtime; these are NOT
    declared in ``observation_space`` (matches dmc — they are runtime-only and
    filtered by MultiEncoder's ``excluded`` set),
  * ``observation_space`` / ``action_space`` are *old* ``gym.spaces`` objects,
  * action_repeat is consumed *inside* this adapter's step (our wrapper owns the
    repeat loop), exactly like ``DeepMindControl`` — ``config.action_repeat`` is
    only used by dreamer.py for counter/time_limit accounting.

Episode truncation (180s) is owned by ``wrappers.TimeLimit`` in the make_env
chain (decision #22), so the wrapper's internal step cap is disabled here.
"""
import os
import sys

import gym  # old gym (0.18) — same as envs/dmc.py and envs/wrappers.py
import numpy as np

# Make the project package importable regardless of CWD.
# vendor/dreamerv3-torch/envs/f1tenth.py -> up 3 = project root.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dreamer_f1tenth.envs.f1tenth_env import (  # noqa: E402
    F110GymnasiumWrapper,
    NUM_BEAMS,
    S_MIN,
    S_MAX,
    V_MIN,
    V_MAX,
)

# TimeLimit (wrappers.TimeLimit) owns truncation; disable the wrapper's own cap.
_DISABLED_STEP_CAP = 10 ** 9


class F1Tenth:
    """dreamerv3-internal adapter around F110GymnasiumWrapper."""

    metadata = {}

    def __init__(self, task, action_repeat=2, seed=0, v_max=None):
        # task == trackname, e.g. "map_easy3" / "Oschersleben" (decision #25).
        # Keep ctor args for pickling (parallel.py cloudpickle-s the env to each
        # worker process; F110Env itself is not picklable). __getstate__/__setstate__
        # send only these args and re-create the env in the subprocess (EzPickle-style).
        # v_max: forward-speed upper bound (m/s) for action_space.high. None -> module
        # default V_MAX (20.0, unchanged Dreamer behavior). Set <20 for Diffuser speed-cap
        # data-collection policies: the cap is enforced via action_space.high, which
        # NormalizeActions reads to map policy [-1,1] -> [V_MIN, v_max], so the policy can
        # never command >v_max. Inner F110GymnasiumWrapper / vehicle dynamics are unchanged
        # (action bound only; the wrapper's own clip to [V_MIN, V_MAX] is a harmless
        # upper safety that <=v_max commands pass through).
        self._task = task
        self._action_repeat = action_repeat
        self._seed = seed
        self._v_max = float(v_max) if v_max is not None else V_MAX
        self._env = F110GymnasiumWrapper(
            trackname=task,
            action_repeat=action_repeat,
            max_episode_steps=_DISABLED_STEP_CAP,
            seed=seed,
        )
        self.reward_range = [-np.inf, np.inf]

    def __getstate__(self):
        # Only ctor args travel to the worker process (envs=N parallel collection).
        return {
            "_task": self._task,
            "_action_repeat": self._action_repeat,
            "_seed": self._seed,
            "_v_max": self._v_max,
        }

    def __setstate__(self, state):
        # Re-create the (non-picklable) F110Env inside the worker process.
        # state.get keeps backward-compat with pickles written before v_max existed.
        self.__init__(
            state["_task"],
            state["_action_repeat"],
            state["_seed"],
            v_max=state.get("_v_max"),
        )

    @property
    def observation_space(self):
        # Only encoder inputs are declared (lidar -> 1D conv, state -> mlp).
        # is_first/is_terminal/is_last are runtime-only (cf. dmc), filtered by
        # MultiEncoder's excluded set.
        return gym.spaces.Dict({
            "lidar": gym.spaces.Box(0.0, 1.0, (NUM_BEAMS,), dtype=np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
        })

    @property
    def action_space(self):
        # Raw scale [steer, speed]; NormalizeActions([-1,1]) is applied in the
        # make_env chain (decision #22). Forward-speed upper bound = self._v_max
        # (default V_MAX=20.0; <20 for Diffuser speed-cap policies). steer bounds and
        # V_MIN (reverse) unchanged — cap limits forward speed only.
        return gym.spaces.Box(
            low=np.array([S_MIN, V_MIN], dtype=np.float32),
            high=np.array([S_MAX, self._v_max], dtype=np.float32),
            dtype=np.float32,
        )

    def _to_internal(self, obs):
        # gymnasium obs (OrderedDict, includes is_first/is_terminal/is_last) is
        # already in the dreamerv3 runtime shape; bools stay bool.
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        return self._to_internal(obs), float(reward), done, info

    def reset(self):
        obs, _info = self._env.reset()
        return self._to_internal(obs)

    def close(self):
        self._env.close()
