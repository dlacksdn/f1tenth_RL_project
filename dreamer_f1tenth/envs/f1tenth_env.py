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
_MAPS_DIR = os.path.join(_PROJECT_ROOT, "maps")  # centerline csv (implementation/002)

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

# Divergence guard (f110 ST dynamics numerical blow-up). A velocity above this is
# physically impossible (v_max=20) -> sim diverged: terminate + sanitize obs.
_VEL_DIVERGE = 1.0e3          # |vel| beyond this = diverged sim
_STATE_DIVERGE = 1.0e3        # nan_to_num posinf replacement for raw state
_STATE_CLIP = 10.0            # final normalized-state clamp (bounds encoder input)

# L_track measured (implementation/002 §2, decision #1 in 004).
TRACK_CONFIGS = {
    "map_easy3": {
        "map_path": os.path.join(_PKG_MAPS, "map_easy3"),  # f110_env appends .yaml
        "map_ext": ".png",
        # ★ on-track green-ribbon pose (implementation/008). 옛 (8.620,11.860,2.356)은
        # 트랙 밖이라 reset이 개방영역에서 시작했음 → GF DNF 원인. max-clearance 2.37m.
        "default_pose": np.array([1.02, -14.66, -2.819842], dtype=np.float32),
        "L_track": 100.57,  # green-ribbon centerline (옛 117.22는 바깥 영역 기준 무효)
        "centerline_csv": os.path.join(_MAPS_DIR, "map_easy3_centerline.csv"),
    },
    "Oschersleben": {
        "map_path": os.path.join(_PKG_MAPS, "Oschersleben"),
        "map_ext": ".png",
        "default_pose": np.array([0.0702245, 0.3002981, 2.79787], dtype=np.float32),
        "L_track": 275.18,  # corrected ribbon centerline (옛 312.61은 바깥 루프 무효)
        "centerline_csv": os.path.join(_MAPS_DIR, "Oschersleben_centerline.csv"),
    },
}

# reverse_guard (v3 §3 1-4, §4-3 line 318, decision #8/#24).
# centerline tangent · vehicle world-frame velocity < 0 (후진) 가 연속
# REVERSE_COUNTER_LIMIT env step 지속 시 terminated, cause='reverse'.
REVERSE_COUNTER_LIMIT = 50  # 50 env step = 1s @ action_repeat=2 (50 env step/s)


def _load_centerline(csv_path):
    """centerline csv (header s,x,y,tx,ty) → (xy (N,2) f32, tangent (N,2) f32)."""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(
            f"centerline csv {csv_path!r} expected (N,5) s,x,y,tx,ty; got {arr.shape}"
        )
    xy = np.ascontiguousarray(arr[:, 1:3])       # (N,2)
    tangent = np.ascontiguousarray(arr[:, 3:5])  # (N,2) unit tangent
    return xy, tangent


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

        # centerline for reverse_guard (v3 §3 1-4). xy (N,2), unit tangent (N,2).
        self._centerline_xy, self._centerline_tangent = _load_centerline(
            cfg["centerline_csv"]
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
        self._reverse_counter = 0
        self._raw_obs = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_obs(self, raw, is_first=False, is_terminal=False, is_last=False):
        lidar = np.asarray(raw["scans"][0], dtype=np.float64)
        # f110 ST dynamics can numerically diverge (inf/huge) -> non-finite obs
        # poisons the replay buffer (encoder overflow -> NaN logit). Sanitize so
        # obs is ALWAYS finite/bounded; the divergence is terminated in step().
        lidar = np.nan_to_num(lidar, nan=LIDAR_MAX, posinf=LIDAR_MAX, neginf=0.0)
        lidar = (np.clip(lidar, 0.0, LIDAR_MAX) / LIDAR_MAX).astype(np.float32)
        if lidar.shape[0] != NUM_BEAMS:
            buf = np.ones((NUM_BEAMS,), dtype=np.float32)
            n = min(lidar.shape[0], NUM_BEAMS)
            buf[:n] = lidar[:n]
            lidar = buf

        vel_x = float(raw["linear_vels_x"][0])
        vel_y = float(raw["linear_vels_y"][0])
        ang_z = float(raw["ang_vels_z"][0])
        state_raw = np.nan_to_num(
            np.array([vel_x, vel_y, ang_z, self._prev_steer, self._prev_speed],
                     dtype=np.float64),
            nan=0.0, posinf=_STATE_DIVERGE, neginf=-_STATE_DIVERGE,
        )
        state = np.clip(state_raw / _STATE_SCALE, -_STATE_CLIP, _STATE_CLIP)

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
        self._reverse_counter = 0
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

        # reverse_guard (v3 §3 1-4, decision #8/#24): centerline tangent · world-frame
        # velocity < 0 (후진) 가 REVERSE_COUNTER_LIMIT env step 연속 지속 시 종료.
        # dot ≥ 0 (전진·정지) 이면 카운터 reset.
        pos = np.array([raw["poses_x"][0], raw["poses_y"][0]], dtype=np.float32)
        yaw = float(raw["poses_theta"][0])
        vel_x = float(raw["linear_vels_x"][0])  # body-frame longitudinal
        vel_y = float(raw["linear_vels_y"][0])  # body-frame lateral (#27 patch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        vel_world = np.array(
            [vel_x * cos_y - vel_y * sin_y, vel_x * sin_y + vel_y * cos_y],
            dtype=np.float32,
        )
        closest_idx = int(np.argmin(((self._centerline_xy - pos) ** 2).sum(axis=1)))
        dot = float(vel_world @ self._centerline_tangent[closest_idx])
        # vel_x<0 (참 후진, body-frame longitudinal state[3]<0) AND dot<0 동시 충족 시만 카운트.
        # vel_x<0 gate는 centerline 오정합(implementation/007: Oschersleben centerline mis-
        # registration) 대비 robust 안전장치 — GF 등 전진(vel_x>0) 정책의 false reverse 방지.
        # v3 option A의 centerline tangent·velocity dot 조건은 유지(게이팅만 추가).
        if vel_x < 0.0 and dot < 0.0:
            self._reverse_counter += 1
        else:
            self._reverse_counter = 0

        # Divergence guard: f110 ST dynamics can numerically blow up (inf/huge
        # vel) under sustained valid commands -> non-finite obs would poison the
        # replay buffer (encoder overflow -> NaN logit, training crash). Detect
        # and terminate as highest priority; _build_obs sanitizes the obs.
        diverged = (
            not np.isfinite([vel_x, vel_y, float(raw["ang_vels_z"][0]),
                             pos[0], pos[1], yaw]).all()
            or not np.isfinite(raw["scans"][0]).all()
            or abs(vel_x) > _VEL_DIVERGE
            or abs(vel_y) > _VEL_DIVERGE
        )

        # Termination priority: diverged > collision > reverse > lap_complete > timeout (#24).
        terminated = False
        truncated = False
        cause = None
        is_terminal = False
        is_last = False
        if diverged:
            terminated = True
            cause = "diverged"
            is_terminal = True
        elif collision:
            terminated = True
            cause = "collision"
            is_terminal = True
        elif self._reverse_counter >= REVERSE_COUNTER_LIMIT:
            terminated = True
            cause = "reverse"
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
            "reverse_counter": self._reverse_counter,
            "trackname": self.trackname,
        }
        self._raw_obs = raw
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
