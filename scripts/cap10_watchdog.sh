#!/usr/bin/env bash
# cap10_watchdog.sh — Diffuser 데이터용 cap-10(V_MAX=10) Oschersleben 정책 학습 + crash 자동 resume.
#
# cap5_watchdog.sh / cap15_watchdog.sh 와 동일(검증된 패턴 + bracket-pgrep). 차이는 V_MAX=10 + 전용 logdir.
# 목적: 중간 tier baseline 정책 제작. cap-5(2랩~107s, step_25k 채택) 와 cap-15(2랩~37s, step_105k)
#       사이의 중간 속도대(예상 2랩 ~60-70s) behavior policy.
# Dreamer 코어/차량 물리 무변경(V_MAX=backward-compat config 파라미터, 기본20).
#
# 운영 방침: cap-5/cap-15 가 학습 중 진동(빠른 라인 시도→2랩째 크래시)했으나, deterministic 봉우리
#   스냅샷(cap-5=step_25k, cap-15=step_105k)으로 baseline 확보에 성공 = proven 패턴. cap-10도 동일
#   레시피(joint 0.3)로 진행하고, 완주 봉우리 확보 후 수동 종료 + eval_gate(--v_max 10) 채택.
#   (joint 제거 실험은 미검증이라 baseline 데이터 확보 우선 → 보류.)
#
# 사용 (detached):
#   setsid nohup bash /home/dlacksdn/f1tenth_RL_project/scripts/cap10_watchdog.sh \
#     > /home/dlacksdn/f1tenth_RL_project/runs/cap10_oschersleben/watchdog.log 2>&1 < /dev/null &
set -u

PROJ=/home/dlacksdn/f1tenth_RL_project
VENDOR="$PROJ/vendor/dreamerv3-torch"

TASK=f1tenth_Oschersleben
LOGDIR="$PROJ/runs/cap10_oschersleben"
TARGET_STEP="${1:-300000}"
V_MAX=10.0

# cap-5/cap-15와 동일 warm-load 운영 파라미터(proven). v_max만 변경.
WARM_CKPT="$PROJ/runs/stage1_map_easy3/latest.pt"
JOINT_DIR="$PROJ/runs/stage1_map_easy3/train_eps"
WARM_LR_SCALE=0.5
JOINT_RATIO=0.3

METRICS="$LOGDIR/metrics.jsonl"
CHECK_INTERVAL=120
GRACE=90
CONFIRM=15

mkdir -p "$LOGDIR"
log(){ echo "[cap10-watchdog $(date '+%F %T')] $*"; }

# ★ bracket-trick `[d]reamer`: pgrep -f 가 같은 문자열 든 셸 명령까지 매칭하는 오탐 방지.
is_alive(){ pgrep -f "[d]reamer\.py.*cap10_oschersleben" >/dev/null 2>&1; }
other_dreamer(){ pgrep -af "[d]reamer\.py.*--envs 8" 2>/dev/null | grep -v "cap10_oschersleben"; }

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
  log "cap10 시작/resume (task=$TASK v_max=$V_MAX logdir=$LOGDIR)"
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
  log "중단: cap10 외 dreamer 학습 생존 → GPU 점유 중, OOM 위험. 종료."
  other_dreamer | while read -r l; do log "  ↳ $l"; done
  exit 1
fi
if [ ! -f "$WARM_CKPT" ]; then log "중단: warm_load_ckpt 부재 → $WARM_CKPT"; exit 1; fi
if [ ! -d "$JOINT_DIR" ] || [ -z "$(ls -A "$JOINT_DIR"/*.npz 2>/dev/null)" ]; then
  log "중단: joint_replay_dir 에 episode(.npz) 없음 → $JOINT_DIR"; exit 1
fi

log "cap10-watchdog 가동. target=$TARGET_STEP v_max=$V_MAX warm=$WARM_CKPT joint=$JOINT_RATIO interval=${CHECK_INTERVAL}s"

while true; do
  if is_alive; then sleep "$CHECK_INTERVAL"; continue; fi
  log "프로세스 미검출(1차). ${CONFIRM}s 후 재확인."
  sleep "$CONFIRM"
  if is_alive; then log "재확인 결과 생존 → 오탐, 감시 계속."; continue; fi
  st="$(last_step)"
  log "프로세스 사망 확정. last_step=$st (target=$TARGET_STEP)"
  if [ "$st" -ge "$TARGET_STEP" ]; then log "목표 step 도달 → cap10 완료. watchdog 종료."; exit 0; fi
  log "목표 미달 → 시작/crash resume."
  start_train
  sleep "$GRACE"
done
