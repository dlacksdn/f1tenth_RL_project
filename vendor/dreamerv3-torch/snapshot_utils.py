"""A-1 snapshot 유틸 (020 §2, 019 §2; 008 snapshot design).

오케스트레이션 계층 — world model/알고리즘 HP 무관. dreamer.py main()에서 호출.
2종 스냅샷:
  - interval: eval_every trigger마다 full ckpt를 step_{N}k.pt로 별도 보존(A15, 덮어쓰기 X).
  - diversity: 학습 중 eval episode의 per-lap lap_time_s를 (0,T] n_bins 등분 bin에 큐레이션,
    bin별 best(최단 lap_time)만 policy_lap{X}s_step{Y}k.pt로 보존(A14).

partial state_dict(inference-only, 008 §2-2 / 020 §3):
  _wm.*(world model 전체) + _task_behavior.actor.*(actor)만. optimizer/value/_slow_value 제외.
  B-2: agent.state_dict()에 world model이 _wm.* 와 _task_behavior._world_model.* 두 경로로 중복
  (공유 텐서)이나, 추론엔 _wm.*만으로 충분(공유).
"""
import glob
import os
import pathlib

import numpy as np
import torch

# 추론에 필요한 키 prefix (rollout 생성: world model heads 전부 + actor).
_INFERENCE_PREFIXES = ("_wm.", "_task_behavior.actor.")


def inference_state_dict(agent):
    """추론 전용 partial state_dict: _wm.* + _task_behavior.actor.* 만 추림."""
    full = agent.state_dict()
    return {
        k: v for k, v in full.items()
        if any(k.startswith(p) for p in _INFERENCE_PREFIXES)
    }


def save_inference_only(agent, path):
    """partial(inference-only) state_dict를 path에 저장(~50MB, optimizer 제외)."""
    torch.save({"agent_state_dict": inference_state_dict(agent)}, str(path))
    return path


def lap_time_bin(lap_time_s, bin_width, lap_max):
    """lap_time_s를 bin_width 고정 폭 bin의 **상한 label**로 분류.

    사용자 결정(2026-05-22): 10초 고정 폭, 상한 110, 110/100/90/…/10 구간별 1개.
    경계 우상향 폐구간 (lo, hi] → 상한 label: (0,10]→10, (10,20]→20, …, (100,110]→110.
    0 이하 또는 lap_max 초과는 None(미저장). (008 트랙별 5등분을 대체)
    """
    if lap_time_s is None or lap_time_s <= 0.0 or lap_time_s > lap_max:
        return None
    idx = int(np.ceil(lap_time_s / bin_width))
    return int(idx * bin_width)   # 구간 상한 label(정수)


def collect_lap_times_from_episode(episode):
    """episode dict(np arrays)에서 per-lap lap_time_s 리스트 추출.

    log_lap_time_s > 0 인 step이 lap 완주 시점(env가 lap 증가 step에만 값 주입, 020 §1).
    """
    if "log_lap_time_s" not in episode:
        return []
    arr = np.asarray(episode["log_lap_time_s"]).reshape(-1)
    return [float(v) for v in arr if v > 0.0]


def collect_eval_lap_times(evaldir, n_recent):
    """evaldir에서 최근 n_recent개 npz의 per-lap lap_time_s를 모은다(이번 eval 라운드)."""
    files = sorted(
        glob.glob(os.path.join(str(evaldir), "*.npz")),
        key=os.path.getmtime,
    )
    if n_recent:
        files = files[-int(n_recent):]
    laps = []
    for f in files:
        try:
            with np.load(f) as data:
                laps.extend(collect_lap_times_from_episode(data))
        except Exception:
            continue
    return laps


def pack_snapshot_state(snapshot_bins, snapshot_best):
    """checkpoint 직렬화용 snapshot 상태 dict (B-1, 021 §6).

    dreamer.py save 블록의 items_to_save["snapshot_state"]로 들어간다. 평범한 dict라
    torch.save로 그대로 직렬화. bins={label:{lap_time,path}}, best={lap_time,path}.
    """
    return {"bins": snapshot_bins, "best": snapshot_best}


def restore_snapshot_state(checkpoint):
    """checkpoint에서 (snapshot_bins, snapshot_best) 복원 (B-1, 021 §6).

    watchdog resume 시 메모리 {} 리셋으로 디스크 기존 policy_* 파일을 모른 채 재저장
    → step suffix 다른 중복 누적. 이를 막기 위해 checkpoint["snapshot_state"]를 복원.
    하위호환: "snapshot_state" 키 부재(구 ckpt) 시 ({}, {}). 항상 새 dict 반환(원본 미공유).
    """
    state = checkpoint.get("snapshot_state") if isinstance(checkpoint, dict) else None
    if not state:
        return {}, {}
    return dict(state.get("bins", {})), dict(state.get("best", {}))


def save_interval_snapshot(items_to_save, logdir, step_k, keep=True):
    """latest.pt 저장 직후 full ckpt를 step_{step_k}k.pt로 별도 복사(덮어쓰기 X, A15)."""
    if not keep:
        return None
    path = pathlib.Path(logdir) / f"step_{int(step_k)}k.pt"
    torch.save(items_to_save, str(path))
    return path


def resolve_track_value(value, trackname):
    """트랙별 dict면 trackname(대소문자 무관) 조회, 스칼라면 그대로, None이면 None.

    map_easy3=(1s,20s) / Oschersleben=(10s,110s)처럼 트랙별 bin 폭·상한을 분리(사용자
    결정 2026-05-22). yaml 소문자 키 vs 'Oschersleben' 대문자 trackname robust.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        return float(value)
    for key in (trackname, trackname.lower()):
        if key in value:
            return float(value[key])
    return None


def _unlink_if(old_path, keep_path):
    """old_path가 keep_path와 다르면 삭제(bin/best당 1개 유지용)."""
    if not old_path:
        return
    old = pathlib.Path(old_path)
    if old.exists() and old != pathlib.Path(keep_path):
        try:
            old.unlink()
        except OSError:
            pass


def update_diversity_snapshots(agent, lap_times, bins_state, best_state, logdir,
                               bin_width, lap_max, step_k):
    """lap_times로 ① 트랙별 bin best ② run best(트랙별) 두 산출물을 갱신 저장.

    사용자 결정(2026-05-22):
      ① diversity: bin_width 고정 폭 구간(상한 lap_max)마다 최단 lap 1개.
         파일 `policy_lap{X:.1f}s_step{Y}k.pt`, bin당 1개(더 빠르면 교체+옛 파일 삭제).
      ② run best: 이 run(=logdir, 트랙별 process)의 최단 lap policy 1개를 계속 갱신.
         파일 `policy_best_lap{X:.1f}s_step{Y}k.pt`.
         B-2(021 §6): "run best"는 process/logdir 단위(트랙별)이지 전 트랙 통합 best가
         아니다. 트랙 전환(Stage2)은 별도 logdir라 snapshot이 이미 독립 — 정상.

    bins_state: {label(상한): {"lap_time","path"}}, best_state: {"lap_time","path"}
    (둘 다 호출 간 유지되는 mutable 누적 상태; B-1로 checkpoint에 persist). 반환: 저장 path 리스트.
    """
    logdir = pathlib.Path(logdir)
    saved = []
    for lt in lap_times:
        label = lap_time_bin(lt, bin_width, lap_max)
        if label is None:
            continue  # 0 이하 / lap_max 초과 → 미저장
        # ① 10초 bin best
        prev = bins_state.get(label)
        if prev is None or lt < prev["lap_time"]:
            path = logdir / f"policy_lap{lt:.1f}s_step{int(step_k)}k.pt"
            save_inference_only(agent, path)
            if prev is not None:
                _unlink_if(prev["path"], path)
            bins_state[label] = {"lap_time": lt, "path": str(path)}
            saved.append(path)
        # ② run best(트랙별, 계속 갱신) — 이 run(logdir) 단위 최단 lap (B-2).
        if not best_state or lt < best_state.get("lap_time", float("inf")):
            bpath = logdir / f"policy_best_lap{lt:.1f}s_step{int(step_k)}k.pt"
            save_inference_only(agent, bpath)
            _unlink_if(best_state.get("path"), bpath)
            best_state.clear()
            best_state.update({"lap_time": lt, "path": str(bpath)})
            saved.append(bpath)
    return saved
