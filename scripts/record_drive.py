#!/usr/bin/env python3
"""학습된 정책의 주행을 mp4로 녹화한다 (watch_drive.py 시연 로직 + 프레임 캡처).

설계 원칙 (watch_drive.py 와 동일)
- ★ 환경/차체/맵/물리 무수정: make_env 체인을 그대로 쓰고 F110Env.render()만 호출.
- ★ 진행 중 학습 무방해: ckpt를 /tmp 스냅샷으로 복사 후 CPU 추론.

녹화 추가분
- RecordDamy: step thunk 실행 뒤 render() → pyglet GL 버퍼를 캡처해 imageio writer에 append.
- 카메라: 첫 프레임에서 map_points min/max에 윈도우 종횡비 보정 + 여백을 줘 맵 전체를
  타이트하게 fit (차 추적 안 함 — on_draw가 고정 ortho라 1회 설정으로 유지).
- lap time 텍스트: 기존 score_label(world y=-800px)은 카메라 fit 시 화면 밖이 될 수 있어
  의존하지 않고, f110.current_obs의 lap_time/count를 PIL로 고정 픽셀 크기 오버레이(가독성 보장).

사용:
  cd /home/dlacksdn/f1tenth_RL_project && source .venv/bin/activate
  python scripts/record_drive.py --logdir runs/stage1_map_easy3 --task f1tenth_map_easy3 \
      --best --episodes 3 --out _thinking/KEEP/map_easy3_best_drive.mp4
"""
import argparse
import collections
import functools
import os
import pathlib
import shutil
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = PROJECT_ROOT / "vendor" / "dreamerv3-torch"
sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
import pyglet  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import tools  # noqa: E402
from dreamer import Dreamer, make_env  # noqa: E402
from parallel import Damy  # noqa: E402
from f110_gym.envs.f110_env import F110Env, WINDOW_W, WINDOW_H  # noqa: E402

# 환경 스텝당 sim 시간 = timestep(0.01) * action_repeat(2) = 0.02s → 실시간 50 fps.
RECORD_FPS = 50
CAMERA_MARGIN = 0.08  # 맵 bbox 둘레 여백 비율 (과도한 줌아웃 금지: 8%)


def build_config(task_override=None, device="cpu"):
    """dreamer.py __main__ 의 config 조립을 복제 (defaults + f1tenth)."""
    raw = yaml.YAML(typ="safe").load((VENDOR / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    defaults = {}
    for name in ["defaults", "f1tenth"]:
        recursive_update(defaults, raw[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args([])

    config.device = device          # 학습 정지 상태 → GPU 사용 가능
    config.precision = 32           # 추론은 fp32로 (amp 비활성, dtype 안정)
    config.envs = 1
    config.parallel = False
    if task_override:
        config.task = task_override
    return config


def find_f110(env):
    """make_env 체인을 .env / ._env 로 따라 내려가 내부 F110Env 를 찾는다."""
    e, seen = env, set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, F110Env):
            return e
        e = getattr(e, "env", None) or getattr(e, "_env", None)
    raise RuntimeError("체인에서 F110Env 를 못 찾음 (render 대상 없음)")


def fit_camera_to_map(renderer):
    """renderer.left/right/bottom/top 을 map_points 전체에 타이트하게 맞춘다.

    map_points 는 (N,3) 의 50px/m 스케일 좌표. 윈도우 종횡비(WINDOW_W/H)에 맞춰
    box 를 보정해 맵이 늘어나지 않게 하고, 둘레에 CAMERA_MARGIN 여백만 준다.
    """
    pts = renderer.map_points
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    w = (xmax - xmin) * (1.0 + 2.0 * CAMERA_MARGIN)
    h = (ymax - ymin) * (1.0 + 2.0 * CAMERA_MARGIN)
    win_aspect = WINDOW_W / WINDOW_H
    box_aspect = w / h
    if box_aspect < win_aspect:
        w = h * win_aspect      # 세로가 기준 → 가로 확장
    else:
        h = w / win_aspect      # 가로가 기준 → 세로 확장
    renderer.left = cx - w / 2.0
    renderer.right = cx + w / 2.0
    renderer.bottom = cy - h / 2.0
    renderer.top = cy + h / 2.0


def _load_font(px):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def capture_frame(f110, ep_idx, font):
    """현재 GL 컬러 버퍼를 캡처해 lap time/episode 텍스트를 오버레이한 RGB 프레임 반환."""
    buf = pyglet.image.get_buffer_manager().get_color_buffer()
    data = buf.get_image_data().get_data("RGB", buf.width * 3)
    frame = np.frombuffer(data, dtype=np.uint8).reshape(buf.height, buf.width, 3)[::-1]
    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img)
    obs = f110.current_obs
    laptime = float(obs["lap_times"][0])
    count = int(obs["lap_counts"][obs["ego_idx"]])
    text = f"Episode {ep_idx + 1}   Lap Time: {laptime:5.2f}s   Lap: {count}"
    # 검은 외곽선 + 흰 글씨로 배경 대비 확보 (맵은 어두운 배경 위 빨간 점).
    x, y = 16, 12
    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


class WindowClosed(Exception):
    """사용자가 렌더 창을 닫았을 때."""


class LapsReached(Exception):
    """목표 랩 수(--laps)를 완주해 녹화를 정상 종료할 때."""


class RecordDamy:
    """Damy 를 감싸 step thunk 실행 직후 render() → 프레임을 writer 에 기록한다."""

    def __init__(self, damy, f110, mode, writer, font, target_laps=0):
        self._damy = damy
        self._f110 = f110
        self._mode = mode
        self._writer = writer
        self._font = font
        self._target_laps = target_laps   # >0 이면 ego가 그만큼 완주하면 정상 종료
        self._fitted = False
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_idx = 0

    def step(self, action):
        promise = self._damy.step(action)

        def wrapped():
            out = promise()
            obs, reward, done, info = out
            self._ep_return += float(reward)
            self._ep_len += 1
            try:
                self._f110.render(mode=self._mode)
                if not self._fitted:
                    # 첫 render 에서 renderer/ map_points 가 생성됨 → 카메라 fit 후 재렌더.
                    fit_camera_to_map(self._f110.renderer)
                    self._f110.render(mode=self._mode)
                    self._fitted = True
                self._writer.append_data(
                    capture_frame(self._f110, self._ep_idx, self._font)
                )
            except (WindowClosed, LapsReached):
                raise
            except Exception as exc:
                raise WindowClosed(str(exc))
            # 목표 랩 완주 시: 마지막 프레임까지 기록한 뒤 정상 종료(writer.close는 finally).
            if self._target_laps > 0:
                fobs = self._f110.current_obs
                if int(fobs["lap_counts"][fobs["ego_idx"]]) >= self._target_laps:
                    print(f"[record] 목표 {self._target_laps}랩 완주 → 녹화 종료 "
                          f"(length={self._ep_len})", flush=True)
                    raise LapsReached(f"{self._target_laps} laps")
            if done:
                self._ep_idx += 1
                print(
                    f"[record] 에피소드 {self._ep_idx}: return={self._ep_return:.2f} "
                    f"length={self._ep_len} cause={info.get('cause')}",
                    flush=True,
                )
                self._ep_return = 0.0
                self._ep_len = 0
            return out

        return wrapped

    def reset(self):
        return self._damy.reset()

    def __getattr__(self, name):
        return getattr(self._damy, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs/stage1_map_easy3")
    ap.add_argument("--ckpt", default=None, help="기본: <logdir>/latest.pt")
    ap.add_argument("--best", action="store_true",
                    help="logdir의 현재 run best(policy_best_lap*.pt)를 자동 선택.")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--laps", type=int, default=0,
                    help=">0 이면 ego가 그만큼 완주하는 즉시 정상 종료(한 에피소드 내 연속 랩 녹화). "
                         "예: --laps 2 → 2바퀴 다 돌면 끝. 기본 0=episodes 만큼 관람.")
    ap.add_argument("--mode", default="human_fast", choices=["human", "human_fast"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="추론 디바이스 (기본: cuda 가용 시 cuda). 학습 정지 상태면 GPU 권장.")
    ap.add_argument("--task", default=None, help="task override (예: f1tenth_Oschersleben)")
    ap.add_argument("--out", required=True, help="출력 mp4 경로 (덮어쓰기 금지 — 존재 시 -1,-2 증분)")
    args = ap.parse_args()

    logdir = (PROJECT_ROOT / args.logdir).resolve() if not os.path.isabs(args.logdir) else pathlib.Path(args.logdir)
    if args.ckpt:
        ckpt_src = pathlib.Path(args.ckpt)
    elif args.best:
        cands = sorted(logdir.glob("policy_best_lap*.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"best 스냅샷 없음: {logdir}/policy_best_lap*.pt")
        ckpt_src = cands[-1]
        print(f"[record] --best 자동 선택: {ckpt_src.name}")
    else:
        ckpt_src = logdir / "latest.pt"
    if not ckpt_src.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_src}")

    # 출력 경로: 덮어쓰기 금지 정책 — 존재하면 -1, -2 ... 로 증분.
    out_path = (PROJECT_ROOT / args.out) if not os.path.isabs(args.out) else pathlib.Path(args.out)
    if out_path.exists():
        stem, suffix, parent = out_path.stem, out_path.suffix, out_path.parent
        n = 1
        while (parent / f"{stem}-{n}{suffix}").exists():
            n += 1
        out_path = parent / f"{stem}-{n}{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[record] 출력: {out_path}")

    snap = pathlib.Path("/tmp/record_drive_snapshot.pt")
    shutil.copy2(ckpt_src, snap)
    print(f"[record] 스냅샷 복사: {ckpt_src} -> {snap}")

    config = build_config(task_override=args.task, device=args.device)
    config.logdir = str(logdir)
    logger = tools.Logger(pathlib.Path("/tmp/record_drive_log"), 0)

    env = make_env(config, "eval", 0)
    f110 = find_f110(env)

    obs_space = env.observation_space
    act_space = env.action_space
    config.num_actions = act_space.shape[0]

    print(f"[record] device={config.device} task={config.task} episodes={args.episodes} "
          f"mode={args.mode} fps={RECORD_FPS}")
    agent = Dreamer(obs_space, act_space, config, logger, dataset=None).to(config.device)
    agent.requires_grad_(False)
    ckpt = torch.load(snap, map_location="cpu")
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[record] partial 로드: missing={len(missing)}키(critic/value 등, 주행 무관) "
              f"unexpected={len(unexpected)}")
    agent.eval()

    font = _load_font(26)
    writer = imageio.get_writer(str(out_path), fps=RECORD_FPS, macro_block_size=1, quality=8)
    renv = RecordDamy(Damy(env), f110, args.mode, writer, font, target_laps=args.laps)

    # --laps 지정 시 한 에피소드 내 연속 랩을 노린다(episodes=1로 두고 랩 달성 시 조기 종료).
    episodes = 1 if args.laps > 0 else args.episodes
    print(f"[record] 모델 로드 완료. 녹화 시작 (episodes={episodes}, target_laps={args.laps}).")
    eval_policy = functools.partial(agent, training=False)
    cache = collections.OrderedDict()
    try:
        tools.simulate(
            eval_policy, [renv], cache, "/tmp/record_drive_eps", logger,
            is_eval=True, episodes=episodes,
        )
        print("[record] 완료: 요청한 에피소드 녹화 종료.", flush=True)
    except LapsReached as exc:
        print(f"[record] 목표 랩 완주로 녹화 정상 종료. ({exc})", flush=True)
    except WindowClosed as exc:
        print(f"[record] 창이 닫혀 녹화를 종료합니다. ({exc})", flush=True)
    except KeyboardInterrupt:
        print("[record] 사용자 중단.", flush=True)
    finally:
        writer.close()
        print(f"[record] mp4 저장 완료: {out_path}", flush=True)
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
