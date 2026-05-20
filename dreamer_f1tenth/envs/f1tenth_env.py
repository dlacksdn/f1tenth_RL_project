"""F110GymnasiumWrapper — gym 0.18 f110-v0 → gymnasium 5-tuple wrapper.

Phase 1-2 scope (v3 §3 1-2, decisions #6/#15/#22/#24, implementation/004):
- gym 0.18 4-tuple → gymnasium 5-tuple (terminated/truncated 분리).
- obs dict 5-key: lidar (1080,) float32 normalized, state (5,) float32 normalized,
  is_first / is_terminal / is_last bool.
- action_space raw scale [s_min, s_max] × [v_min, v_max]. NormalizeActions([-1,1])는 외부 chain (#22).
- terminated 우선순위: collision > reverse(stub, Phase 1-4) > lap_complete > timeout.
- reward는 Phase 1-2에서 0.0 skeleton. Phase 4 §4-3 progress + R_lap 추가.
- max_episode_steps=9000 env step (= 180s @ action_repeat=2 / 50 env step/s)을 wrapper 자체 처리.
- false collision guard (decision #3, dqn.py:167 패턴): reset 직후 첫 step의 collision 1회 무시.
"""
import os
from collections import OrderedDict

import numpy as np
import gymnasium
from gymnasium import spaces

# f110_gym is editable-installed; direct import.
from f110_gym.envs.f110_env import F110Env


# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG_MAPS = os.path.join(_PROJECT_ROOT, "pkg", "src", "pkg", "maps")

# Vehicle/action bounds — f110_env default params.
S_MIN, S_MAX = -0.4189, 0.4189
V_MIN, V_MAX = -5.0, 20.0
LIDAR_MAX = 30.0
NUM_BEAMS = 1080
SIM_TIMESTEP = 0.01           # base physics step
DEFAULT_ACTION_REPEAT = 2     # → env step = 0.02s, 50 env step / s
DEFAULT_MAX_EP_STEPS = 9000   # 180s @ 50 env step / s

# State normalization scales.
# - vel_x / 20  (v3 결정 #15)
# - vel_y / 5   (v3 결정 #15; Phase 1-3 base_classes patch 전엔 항상 0)
# - ang_vel_z / (2π)  (★ Phase 1-2 갱신: implementation/005 §2-1.
#       v3 #15 원안 /π 는 실측(|.| 99%=6.22, max=10.1) 대비 부족 → 99%-ile≤1 위해 /2π 채택.)
# - prev_steer / 0.4189   (= s_max)
# - prev_speed / 20       (= v_max)
_STATE_SCALE = np.array([20.0, 5.0, 2.0 * np.pi, 0.4189, 20.0], dtype=np.float32)

# L_track measured (implementation/002 §2, decision #1 in 004).
TRACK_CONFIGS = {
    "map_easy3": {
        "map_path": os.path.join(_PKG_MAPS, "map_easy3"),  # f110_env appends .yaml
        "map_ext": ".png",
        "default_pose": np.array([8.620, 11.860, 2.356], dtype=np.float32),
        "L_track": 117.22,
    },
    "Oschersleben": {
        "map_path": os.path.join(_PKG_MAPS, "Oschersleben"),
        "map_ext": ".png",
        "default_pose": np.array([0.0702245, 0.3002981, 2.79787], dtype=np.float32),
        "L_track": 312.61,
    },
}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class F110GymnasiumWrapper(gymnasium.Env):
    """Gymnasium-API wrapper around f110_gym:f110-v0."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        trackname: str = "map_easy3",
        action_repeat: int = DEFAULT_ACTION_REPEAT,
        max_episode_steps: int = DEFAULT_MAX_EP_STEPS,
        default_pose=None,
        ignore_first_collision: bool = True,
        seed: int = 12345,
    ):
        super().__init__()
        if trackname not in TRACK_CONFIGS:
            raise ValueError(
                f"Unknown trackname {trackname!r}. Choices: {list(TRACK_CONFIGS)}"
            )
        cfg = TRACK_CONFIGS[trackname]
        self.trackname = trackname
        self.action_repeat = int(action_repeat)
        self.max_episode_steps = int(max_episode_steps)
        self.ignore_first_collision = bool(ignore_first_collision)
        self.L_track = float(cfg["L_track"])
        self._default_pose = (
            np.asarray(default_pose, dtype=np.float32).reshape(3).copy()
            if default_pose is not None
            else cfg["default_pose"].copy()
        )

        self._env = F110Env(
            map=cfg["map_path"],
            map_ext=cfg["map_ext"],
            num_agents=1,
            timestep=SIM_TIMESTEP,
            ego_idx=0,
            seed=seed,
        )

        # gymnasium spaces.
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0.0, 1.0, shape=(NUM_BEAMS,), dtype=np.float32),
            "state": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "is_first": spaces.Discrete(2),
            "is_terminal": spaces.Discrete(2),
            "is_last": spaces.Discrete(2),
        })
        self.action_space = spaces.Box(
            low=np.array([S_MIN, V_MIN], dtype=np.float32),
            high=np.array([S_MAX, V_MAX], dtype=np.float32),
            shape=(2,), dtype=np.float32,
        )

        # internal state
        self._env_step = 0
        self._prev_steer = 0.0
        self._prev_speed = 0.0
        self._first_step_done = False
        self._raw_obs = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_obs(self, raw, is_first=False, is_terminal=False, is_last=False):
        lidar = np.asarray(raw["scans"][0], dtype=np.float32)
        lidar = np.clip(lidar, 0.0, LIDAR_MAX) / LIDAR_MAX
        if lidar.shape[0] != NUM_BEAMS:
            buf = np.ones((NUM_BEAMS,), dtype=np.float32)
            n = min(lidar.shape[0], NUM_BEAMS)
            buf[:n] = lidar[:n]
            lidar = buf

        vel_x = float(raw["linear_vels_x"][0])
        vel_y = float(raw["linear_vels_y"][0])
        ang_z = float(raw["ang_vels_z"][0])
        state_raw = np.array(
            [vel_x, vel_y, ang_z, self._prev_steer, self._prev_speed],
            dtype=np.float32,
        )
        state = state_raw / _STATE_SCALE

        return OrderedDict([
            ("lidar", lidar),
            ("state", state.astype(np.float32)),
            ("is_first", bool(is_first)),
            ("is_terminal", bool(is_terminal)),
            ("is_last", bool(is_last)),
        ])

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "pose" in options:
            pose = np.asarray(options["pose"], dtype=np.float32).reshape(3)
        else:
            pose = self._default_pose.copy()

        poses = pose.reshape(1, 3)
        raw, _r, _done, _info = self._env.reset(poses)

        self._env_step = 0
        self._prev_steer = 0.0
        self._prev_speed = 0.0
        self._first_step_done = False
        self._raw_obs = raw

        obs = self._build_obs(raw, is_first=True, is_terminal=False, is_last=False)
        info = {"cause": None, "trackname": self.trackname, "env_step": 0}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        steer = float(np.clip(action[0], S_MIN, S_MAX))
        speed = float(np.clip(action[1], V_MIN, V_MAX))

        action_2d = np.array([[steer, speed]], dtype=np.float64)

        raw = self._raw_obs
        # action_repeat: sub-step until done from base env or repeat exhausted.
        for _ in range(self.action_repeat):
            raw, _r, base_done, _i = self._env.step(action_2d)
            if base_done:
                break

        self._env_step += 1

        collision = bool(raw["collisions"][0])
        # False collision guard on the very first step after reset (decision #3).
        if collision and self.ignore_first_collision and not self._first_step_done:
            collision = False
        self._first_step_done = True

        lap_count = int(raw["lap_counts"][0])

        # Termination priority: collision > reverse(stub) > lap_complete > timeout.
        terminated = False
        truncated = False
        cause = None
        is_terminal = False
        is_last = False
        if collision:
            terminated = True
            cause = "collision"
            is_terminal = True
        elif lap_count >= 2:
            terminated = True
            cause = "lap_complete"
            is_last = True
        elif self._env_step >= self.max_episode_steps:
            truncated = True
            cause = "timeout"

        # prev_* reflect the action just applied; visible to NEXT obs build.
        self._prev_steer = steer
        self._prev_speed = speed

        obs = self._build_obs(raw, is_first=False, is_terminal=is_terminal, is_last=is_last)

        # Phase 1-2 skeleton: reward=0.0. Phase 4 adds §4-3 progress + R_lap.
        reward = 0.0

        info = {
            "cause": cause,
            "collision_raw": bool(raw["collisions"][0]),
            "lap_count": lap_count,
            "env_step": self._env_step,
            "trackname": self.trackname,
        }
        self._raw_obs = raw
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
