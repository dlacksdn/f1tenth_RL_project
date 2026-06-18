"""F110GymnasiumWrapper — gym 0.18 f110-v0 → gymnasium 5-tuple wrapper.

Phase 1-2 scope (v3 §3 1-2, decisions #6/#15/#22/#24, implementation/004):
- gym 0.18 4-tuple → gymnasium 5-tuple (terminated/truncated 분리).
- obs dict 5-key: lidar (1080,) float32 normalized, state (5,) float32 normalized,
  is_first / is_terminal / is_last bool.
- action_space raw scale [s_min, s_max] × [v_min, v_max]. NormalizeActions([-1,1])는 외부 chain (#22).
- terminated 우선순위: diverged > collision > reverse > lap_complete > timeout (#24).
- reward (Phase 4, §4-3·4-4 + 009 결정 A): arclength windowed-progress(clip 0~0.5m)
  + R_lap(Map=25/Osch=100) + diverged/collision/reverse 페널티(-10). lap 판정은
  centerline arclength wrap 기반(f110 lap_count 미사용).
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
# R_lap: per-lap bonus (planning/005 §4-3 산술표, Map/Track 분리 — 통일 금지).
#   Map Easy3=25 (lap당 progress 총합 ≈100.57의 ~25%), Oschersleben=100 (≈275의 ~36%).
TRACK_CONFIGS = {
    "map_easy3": {
        "map_path": os.path.join(_PKG_MAPS, "map_easy3"),  # f110_env appends .yaml
        "map_ext": ".png",
        # ★ on-track green-ribbon pose (implementation/008). 옛 (8.620,11.860,2.356)은
        # 트랙 밖이라 reset이 개방영역에서 시작했음 → GF DNF 원인. max-clearance 2.37m.
        "default_pose": np.array([1.02, -14.66, -2.819842], dtype=np.float32),
        "L_track": 100.57,  # green-ribbon centerline (옛 117.22는 바깥 영역 기준 무효)
        "R_lap": 25.0,
        "centerline_csv": os.path.join(_MAPS_DIR, "map_easy3_centerline.csv"),
    },
    "Oschersleben": {
        "map_path": os.path.join(_PKG_MAPS, "Oschersleben"),
        "map_ext": ".png",
        "default_pose": np.array([0.0702245, 0.3002981, 2.79787], dtype=np.float32),
        "L_track": 275.18,  # corrected ribbon centerline (옛 312.61은 바깥 루프 무효)
        "R_lap": 100.0,
        "centerline_csv": os.path.join(_MAPS_DIR, "Oschersleben_centerline.csv"),
    },
}

# ---------------------------------------------------------------------------
# Reward / arclength-progress config (Phase 4, planning/005 §4-3·4-4 + 009 결정 A).
# 단위: env step (= action_repeat sim step = 0.02s, 50 env step/s).
# ---------------------------------------------------------------------------
ALPHA_PROGRESS = 1.0          # progress reward 계수 (005 §4-1)
PROGRESS_CAP = 0.5            # step-cap clip(arclen_delta, 0, 0.5)m. max v=20×0.02=0.4m+여유 (005 §4-3)
PENALTY_TERMINAL = -10.0      # collision / reverse / diverged 종료 페널티 (005 §4-1, smoke_findings #4)
LAP_TARGET = 2               # 2-lap 완주 시 lap_complete (009 결정 B, completion-only)

# windowed closest-point progress (009 결정 A): 이전 idx ±window 범위에서만 closest 탐색
# → self-intersection 구간의 global-argmin 점프 방지. 거리(m) 기반으로 트랙별 인덱스 환산.
SEARCH_FWD_M = 1.5           # 전방 탐색 범위. max progress 0.4m/step의 ~3.75배 여유
SEARCH_BACK_M = 0.5          # 후방 탐색 범위 (정지/미세후진 대비)

# reverse_guard (v3 §3 1-4, §4-3 line 318, decision #8/#24).
# centerline tangent · vehicle world-frame velocity < 0 (후진) 가 연속
# REVERSE_COUNTER_LIMIT env step 지속 시 terminated, cause='reverse'.
REVERSE_COUNTER_LIMIT = 50  # 50 env step = 1s @ action_repeat=2 (50 env step/s)


def _load_centerline(csv_path):
    """centerline csv (header s,x,y,tx,ty) → (s (N,) f32, xy (N,2) f32, tangent (N,2) f32)."""
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(
            f"centerline csv {csv_path!r} expected (N,5) s,x,y,tx,ty; got {arr.shape}"
        )
    s = np.ascontiguousarray(arr[:, 0])          # (N,) cumulative arclength
    xy = np.ascontiguousarray(arr[:, 1:3])       # (N,2)
    tangent = np.ascontiguousarray(arr[:, 3:5])  # (N,2) unit tangent
    return s, xy, tangent


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
        # 1 wrapper-step 실경과 sim 시간(s). f110 timestep × action_repeat로 도출
        # (0.02 하드코딩 금지, 020 §3-4: action_repeat 변경 시 silent 오류 방지).
        self._dt_wrap = SIM_TIMESTEP * self.action_repeat
        self.L_track = float(cfg["L_track"])
        self.R_lap = float(cfg["R_lap"])
        self._default_pose = (
            np.asarray(default_pose, dtype=np.float32).reshape(3).copy()
            if default_pose is not None
            else cfg["default_pose"].copy()
        )

        # centerline for reverse_guard + arclength progress (v3 §3 1-4, 009 결정 A).
        # s (N,) cumulative arclength, xy (N,2), unit tangent (N,2).
        self._centerline_s, self._centerline_xy, self._centerline_tangent = (
            _load_centerline(cfg["centerline_csv"])
        )
        # windowed closest-point: 거리(m)→인덱스 환산 (트랙별 평균 점 간격 기준).
        n_pts = len(self._centerline_xy)
        ds_mean = self.L_track / max(n_pts, 1)
        self._fwd_window = max(int(np.ceil(SEARCH_FWD_M / ds_mean)), 1)
        self._back_window = max(int(np.ceil(SEARCH_BACK_M / ds_mean)), 1)

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
        # arclength progress tracking (009 결정 A)
        self._closest_idx = 0       # windowed closest-point index (이전 step 기준)
        self._total_arclen = 0.0    # 시작점 기준 누적 진행거리(m, 부호 포함)
        self._lap_count_arc = 0     # arclength wrap 기반 lap 카운트
        self._lap_start_step = 0    # 현재 lap 시작 env_step (per-lap lap_time_s 산출용, 020 §1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    # 진단 log_ 신호 키 집합 (020 §1 경로①). observation_space에는 넣지 않으며
    # (5-key 유지), step/reset 모든 transition에 동일 키를 채워 simulate cache의
    # 키 일관성을 보장한다. tools.simulate는 "log_" 키를 encoder 입력에서 strip
    # (tools.py:167) → world model 무영향. transition=o.copy()로 cache/npz 보존
    # → save_episodes(npz) + log_ TB 합산(tools.py:213-217). 모든 값 float32.
    _LOG_KEYS = (
        "log_lap_time_s",        # per-lap 완주 step에만 그 lap 실경과 sim 시간(s); 그 외 0
        "log_reward_progress",   # A-5(A17) reward component 분리 (step별)
        "log_reward_collision",
        "log_reward_reverse",
        "log_reward_diverged",
        "log_reward_lap",
        "log_lap_count_arc",     # 현재 누적 lap 수(진단; npz per-step 추적용)
        "log_completed",         # lap_complete(2-lap) 종료 step에 1.0; 그 외 0
        # Diffuser P2(planning/005 §5·§6): world pose 진단 채널. raw poses_x/y/theta[0]를
        # log_ 키로 노출 → simulate/run_episode가 encoder 입력에서 strip(tools.py:167,
        # eval_gate.py:183) → WM/물리/reward/판정 무영향. transition=o.copy()로 npz 보존.
        # collect_crash_data.py가 npz의 이 3키를 stack→pose(T,3)로 만든다(env.unwrapped
        # ._raw_obs는 어댑터가 gym.Wrapper 미상속이라 끊김 — 이 채널이 견고한 경로).
        "log_pose_x",            # world x = raw["poses_x"][0]
        "log_pose_y",            # world y = raw["poses_y"][0]
        "log_pose_theta",        # world yaw = raw["poses_theta"][0]
    )

    def _build_obs(self, raw, is_first=False, is_terminal=False, is_last=False,
                   log_fields=None):
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

        obs = OrderedDict([
            ("lidar", lidar),
            ("state", state.astype(np.float32)),
            ("is_first", bool(is_first)),
            ("is_terminal", bool(is_terminal)),
            ("is_last", bool(is_last)),
        ])
        # log_ 진단 키: 기본 0.0, log_fields로 덮어쓰기. 전 transition 동일 키 집합.
        lf = log_fields or {}
        for k in self._LOG_KEYS:
            obs[k] = np.float32(lf.get(k, 0.0))
        return obs

    def _global_closest_idx(self, pos):
        """전역 argmin closest-point index. reset 시 1회 사용."""
        return int(np.argmin(((self._centerline_xy - pos) ** 2).sum(axis=1)))

    def _windowed_closest_idx(self, pos):
        """이전 closest_idx 주변 [-back, +fwd] (인덱스 순환) 에서만 closest 탐색.
        009 결정 A: self-intersection 구간의 global-argmin 점프를 막는다."""
        n = len(self._centerline_xy)
        lo = self._closest_idx - self._back_window
        span = self._back_window + self._fwd_window + 1
        cand = (np.arange(lo, lo + span) % n)              # 순환 인덱스
        d2 = ((self._centerline_xy[cand] - pos) ** 2).sum(axis=1)
        return int(cand[int(np.argmin(d2))])

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

        # arclength progress: start pose의 전역 closest-point를 기준(0)으로 초기화.
        start_pos = np.array([raw["poses_x"][0], raw["poses_y"][0]], dtype=np.float32)
        self._closest_idx = self._global_closest_idx(start_pos)
        self._total_arclen = 0.0
        self._lap_count_arc = 0
        self._lap_start_step = 0

        # is_first transition: log_ 키는 기본 0이되 world pose만 reset 시점 실제값 주입
        # (collect_crash_data.py가 pose(T,3) 첫 행으로 사용; 다른 log_ 키는 0 유지).
        obs = self._build_obs(
            raw, is_first=True, is_terminal=False, is_last=False,
            log_fields={
                "log_pose_x": float(raw["poses_x"][0]),
                "log_pose_y": float(raw["poses_y"][0]),
                "log_pose_theta": float(raw["poses_theta"][0]),
            },
        )
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

        lap_count = int(raw["lap_counts"][0])  # f110 lap (디버그 로깅용; 판정엔 미사용, 009 §2)

        pos = np.array([raw["poses_x"][0], raw["poses_y"][0]], dtype=np.float32)
        yaw = float(raw["poses_theta"][0])
        vel_x = float(raw["linear_vels_x"][0])  # body-frame longitudinal
        vel_y = float(raw["linear_vels_y"][0])  # body-frame lateral (#27 patch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        vel_world = np.array(
            [vel_x * cos_y - vel_y * sin_y, vel_x * sin_y + vel_y * cos_y],
            dtype=np.float32,
        )

        # ---- arclength windowed closest-point progress (009 결정 A) ----------
        # 이전 idx ±window 에서만 탐색 → self-intersection global-argmin 점프 차단.
        prev_idx = self._closest_idx
        new_idx = self._windowed_closest_idx(pos)
        s_prev = float(self._centerline_s[prev_idx])
        s_now = float(self._centerline_s[new_idx])
        raw_delta = s_now - s_prev
        # start/finish 라인 통과 시 s wrap 보정 (닫힌 루프 가정; 큰 점프 = 경계 통과).
        if raw_delta < -self.L_track / 2.0:
            raw_delta += self.L_track       # 순방향 wrap (finish→start)
        elif raw_delta > self.L_track / 2.0:
            raw_delta -= self.L_track       # 역방향 wrap
        self._closest_idx = new_idx
        self._total_arclen += raw_delta     # 시작점 기준 누적 진행거리(부호 포함)
        # lap = 순방향 누적의 high-water-mark (경계 왕복 reward farming 방지 = 방향 가드).
        current_lap = int(self._total_arclen // self.L_track)
        lap_increased = current_lap > self._lap_count_arc
        # per-lap lap_time_s (020 §1·§1-2): lap 증가 step에만 그 lap 실경과 sim 시간.
        # = Δenv_step × DT_WRAP(=SIM_TIMESTEP×action_repeat). 진단 신호, 판정 무영향.
        lap_time_s = 0.0
        if lap_increased:
            self._lap_count_arc = current_lap
            lap_time_s = (self._env_step - self._lap_start_step) * self._dt_wrap
            self._lap_start_step = self._env_step

        # reverse_guard (v3 §3 1-4, decision #8/#24): centerline tangent · world-frame
        # velocity < 0 (후진) 가 REVERSE_COUNTER_LIMIT env step 연속 지속 시 종료.
        # dot ≥ 0 (전진·정지) 이면 카운터 reset. windowed closest_idx 사용(009 §2).
        dot = float(vel_world @ self._centerline_tangent[new_idx])
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

        # ---- reward components (planning/005 §4-1·4-3·4-4, A17 분리 로깅) ------
        # progress: env-step당 arclength 증분(m), step-cap clip(0, 0.5). 후진은 0.
        progress_r = ALPHA_PROGRESS * float(np.clip(raw_delta, 0.0, PROGRESS_CAP))
        collision_r = 0.0
        reverse_r = 0.0
        diverged_r = 0.0
        lap_r = 0.0

        # Termination priority: diverged > collision > reverse > lap_complete > timeout (#24).
        # lap_complete·R_lap·progress 모두 arclength 신호 사용 (f110 lap_count 미사용, 009).
        terminated = False
        truncated = False
        cause = None
        is_terminal = False
        is_last = False
        if diverged:
            terminated = True
            cause = "diverged"
            is_terminal = True
            diverged_r = PENALTY_TERMINAL  # smoke_findings #4: 발산=비정상 실패 → 페널티
        elif collision:
            terminated = True
            cause = "collision"
            is_terminal = True
            collision_r = PENALTY_TERMINAL
        elif self._reverse_counter >= REVERSE_COUNTER_LIMIT:
            terminated = True
            cause = "reverse"
            is_terminal = True
            reverse_r = PENALTY_TERMINAL
        else:
            # 정상 진행: 새 high-water lap 완주 시 R_lap 가산 (방향가드 내장).
            if lap_increased:
                lap_r = self.R_lap
            if self._lap_count_arc >= LAP_TARGET:
                terminated = True
                cause = "lap_complete"
                is_last = True
            elif self._env_step >= self.max_episode_steps:
                truncated = True
                cause = "timeout"

        reward = progress_r + collision_r + reverse_r + diverged_r + lap_r

        # prev_* reflect the action just applied; visible to NEXT obs build.
        self._prev_steer = steer
        self._prev_speed = speed

        # 진단 log_ 신호(020 §1 경로①): obs에 노출 → simulate cache/npz 보존 + TB 합산.
        # reward component(A-5/A17)와 per-lap lap_time(A-1)을 동일 채널로 흡수.
        log_fields = {
            "log_lap_time_s": lap_time_s,
            "log_reward_progress": progress_r,
            "log_reward_collision": collision_r,
            "log_reward_reverse": reverse_r,
            "log_reward_diverged": diverged_r,
            "log_reward_lap": lap_r,
            "log_lap_count_arc": float(self._lap_count_arc),
            "log_completed": 1.0 if cause == "lap_complete" else 0.0,
            # world pose 진단(Diffuser P2): pos/yaw는 위 L328-329에서 이미 계산됨.
            "log_pose_x": float(pos[0]),
            "log_pose_y": float(pos[1]),
            "log_pose_theta": yaw,
        }
        obs = self._build_obs(raw, is_first=False, is_terminal=is_terminal,
                              is_last=is_last, log_fields=log_fields)

        info = {
            "cause": cause,
            "collision_raw": bool(raw["collisions"][0]),
            "lap_count": lap_count,            # f110 raw (디버그용)
            "lap_count_arc": self._lap_count_arc,  # arclength 판정 lap (SSOT)
            "total_arclen": self._total_arclen,
            "arclen_s": s_now,
            "closest_idx": new_idx,
            "env_step": self._env_step,
            "reverse_counter": self._reverse_counter,
            "trackname": self.trackname,
            # A17 reward component 분리 로깅
            "reward_progress": progress_r,
            "reward_collision": collision_r,
            "reward_reverse": reverse_r,
            "reward_diverged": diverged_r,
            "reward_lap": lap_r,
        }
        self._raw_obs = raw
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
