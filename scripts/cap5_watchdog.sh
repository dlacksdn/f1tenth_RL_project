#!/usr/bin/env bash
# cap5_watchdog.sh — Diffuser 데이터용 cap-5(V_MAX=5) Oschersleben 정책 학습 + crash 자동 resume.
#
# cap15_watchdog.sh 와 동일(검증된 패턴 + bracket-pgrep). 차이는 V_MAX=5 + 전용 logdir.
# 목적: slide "약 100초대 behavior policy" = 저속(V_MAX=5)으로 느리게 완주하는 정책 제작
# (cap-15는 ~37s로 너무 빨라 baseline 부적합 — 트랙 평균속도<15라 15는 사실상 무캡).
# Dreamer 코어/차량 물리 무변경(V_MAX=backward-compat config 파라미터, 기본20).
#
# 사용 (detached):
#   setsid nohup bash /home/dlacksdn/f1tenth_RL_project/scripts/cap5_watchdog.sh \
#     > /home/dlacksdn/f1tenth_RL_project/runs/cap5_oschersleben/watchdog.log 2>&1 < /dev/null &
set -u

PROJ=/home/dlacksdn/f1tenth_RL_project
VENDOR="$PROJ/vendor/dreamerv3-torch"

TASK=f1tenth_Oschersleben
LOGDIR="$PROJ/runs/cap5_oschersleben"
TARGET_STEP="${1:-300000}"
V_MAX=5.0

# cap-15와 동일 warm-load 운영 파라미터(proven). v_max만 변경.
# (NOTE: cap-15 학습이 진동했고 joint_replay 혼합이 유력 원인 추정 — cap-5는 일단 proven
#  설정 그대로 가고, joint 제거 실험은 cap-10에서 의도적으로 시도. implementation/031 참조.)
WARM_CKPT="$PROJ/runs/stage1_map_easy3/latest.pt"
JOINT_DIR="$PROJ/runs/stage1_map_easy3/train_eps"
WARM_LR_SCALE=0.5
JOINT_RATIO=0.3

METRICS="$LOGDIR/metrics.jsonl"
CHECK_INTERVAL=120
GRACE=90
CONFIRM=15

mkdir -p "$LOGDIR"
log(){ echo "[cap5-watchdog $(date '+%F %T')] $*"; }

# ★ bracket-trick `[d]reamer`: pgrep -f 가 같은 문자열 든 셸 명령까지 매칭하는 오탐 방지.
is_alive(){ pgrep -f "[d]reamer\.py.*cap5_oschersleben" >/dev/null 2>&1; }
other_dreamer(){ pgrep -af "[d]reamer\.py.*--envs 8" 2>/dev/null | grep -v "cap5_oschersleben"; }

last_step(){
  python3 - "$METRICS" <<'PY' 2>/dev/null
import json, sys
fn = sys.argv[1]; s = 0
try:
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "step" in d:
                    s = max(s, int(d["step"]))
            except Exception:
                pass
except FileNotFoundError:
    pass
print(s)
PY
}

start_train(){
  log "cap5 시작/resume (task=$TASK v_max=$V_MAX logdir=$LOGDIR)"
  cd "$VENDOR" || { log "cd vendor 실패"; return 1; }
  # shellcheck disable=SC1091
  source "$PROJ/.venv/bin/activate"
  setsid nohup python -u dreamer.py --configs f1tenth \
    --task "$TASK" --logdir "$LOGDIR" \
    --v_max "$V_MAX" --steps "$TARGET_STEP" \
    --warm_load_ckpt "$WARM_CKPT" --warm_lr_scale "$WARM_LR_SCALE" \
    --joint_replay_dir "$JOINT_DIR" --joint_replay_ratio "$JOINT_RATIO" \
    --envs 8 --parallel True --log_every 500 \
    >> "$LOGDIR/train.log" 2>&1 < /dev/null &
  local pid=$!
  log "발사 PID=$pid (v_max=$V_MAX warm=$WARM_CKPT lr×$WARM_LR_SCALE joint=$JOINT_RATIO steps=$TARGET_STEP)"
}

# --- 사전 안전 점검 ---
if other_dreamer >/dev/null; then
  log "중단: cap5 외 dreamer 학습 생존 → GPU 점유 중, OOM 위험. 종료."
  other_dreamer | while read -r l; do log "  ↳ $l"; done
  exit 1
fi
if [ ! -f "$WARM_CKPT" ]; then log "중단: warm_load_ckpt 부재 → $WARM_CKPT"; exit 1; fi
if [ ! -d "$JOINT_DIR" ] || [ -z "$(ls -A "$JOINT_DIR"/*.npz 2>/dev/null)" ]; then
  log "중단: joint_replay_dir 에 episode(.npz) 없음 → $JOINT_DIR"; exit 1
fi

log "cap5-watchdog 가동. target=$TARGET_STEP v_max=$V_MAX warm=$WARM_CKPT joint=$JOINT_RATIO interval=${CHECK_INTERVAL}s"

while true; do
  if is_alive; then sleep "$CHECK_INTERVAL"; continue; fi
  log "프로세스 미검출(1차). ${CONFIRM}s 후 재확인."
  sleep "$CONFIRM"
  if is_alive; then log "재확인 결과 생존 → 오탐, 감시 계속."; continue; fi
  st="$(last_step)"
  log "프로세스 사망 확정. last_step=$st (target=$TARGET_STEP)"
  if [ "$st" -ge "$TARGET_STEP" ]; then log "목표 step 도달 → cap5 완료. watchdog 종료."; exit 0; fi
  log "목표 미달 → 시작/crash resume."
  start_train
  sleep "$GRACE"
done
