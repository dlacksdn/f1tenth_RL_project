#!/usr/bin/env bash
# stage2_watchdog.sh — Stage2 Oschersleben fine-tune 시작 + crash 자동 resume 감시.
#
# Stage1 train_watchdog.sh 와 동일 패턴 + warm-load/joint-replay 인자(A-2, 024).
# start_train 은 항상 동일한 풀 커맨드(warm_load_ckpt + joint)를 발사한다:
#   - 첫 실행: runs/stage2_oschersleben/latest.pt 부재 → dreamer 가 _do_warm
#     (Stage1 world model weights만 warm-load, actor/critic/optim fresh, lr×0.5).
#   - crash resume: latest.pt 존재 → dreamer 가 resume 우선(warm 무시, scaled-lr 복원).
#     joint_replay 는 _do_warm 과 독립이라 resume 시에도 계속 적용(024 V4 확인).
#
# 사용 (detached, ★ Stage1 500k 종료·GPU 반환 후에만):
#   cd /home/dlacksdn/f1tenth_RL_project
#   setsid nohup ./scripts/stage2_watchdog.sh \
#     > runs/stage2_oschersleben/watchdog.log 2>&1 < /dev/null &
#
# 인자 (선택):
#   $1 TARGET_STEP  (default 500000 = configs steps=5e5. 조기 만족 시 수동 중단 가능.)
set -u

PROJ=/home/dlacksdn/f1tenth_RL_project
VENDOR="$PROJ/vendor/dreamerv3-torch"

TASK=f1tenth_Oschersleben
LOGDIR="$PROJ/runs/stage2_oschersleben"
TARGET_STEP="${1:-500000}"

# A-2 운영 파라미터 (024 확정: lr×0.5 forgetting 방어, joint 0.3 Stage1 혼합)
WARM_CKPT="$PROJ/runs/stage1_map_easy3/latest.pt"
JOINT_DIR="$PROJ/runs/stage1_map_easy3/train_eps"
WARM_LR_SCALE=0.5
JOINT_RATIO=0.3

METRICS="$LOGDIR/metrics.jsonl"
CHECK_INTERVAL=120
GRACE=90
CONFIRM=15

mkdir -p "$LOGDIR"
log(){ echo "[stage2-watchdog $(date '+%F %T')] $*"; }

# Stage2(Oschersleben) 메인 생존 여부
is_alive(){ pgrep -f "dreamer.py.*--task ${TASK}.*--envs 8" >/dev/null 2>&1; }
# Stage1(map_easy3) 생존 여부 — GPU 반환 확인용
stage1_alive(){ pgrep -f "dreamer.py.*--task f1tenth_map_easy3.*--envs 8" >/dev/null 2>&1; }

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
  log "Stage2 시작/resume (task=$TASK, logdir=$LOGDIR)"
  cd "$VENDOR" || { log "cd vendor 실패"; return 1; }
  # shellcheck disable=SC1091
  source "$PROJ/.venv/bin/activate"
  setsid nohup python -u dreamer.py --configs f1tenth \
    --task "$TASK" --logdir "$LOGDIR" \
    --warm_load_ckpt "$WARM_CKPT" --warm_lr_scale "$WARM_LR_SCALE" \
    --joint_replay_dir "$JOINT_DIR" --joint_replay_ratio "$JOINT_RATIO" \
    --envs 8 --parallel True --log_every 500 \
    >> "$LOGDIR/train.log" 2>&1 < /dev/null &
  local pid=$!
  log "발사 PID=$pid (warm=$WARM_CKPT lr×$WARM_LR_SCALE joint=$JOINT_RATIO)"
}

# --- 사전 안전 점검 (OOM/오타 fail-fast) ---
if stage1_alive; then
  log "중단: Stage1(map_easy3) 아직 생존 → GPU 점유 중. Stage2 동시 실행은 OOM. 종료."
  exit 1
fi
if [ ! -f "$WARM_CKPT" ]; then
  log "중단: warm_load_ckpt 부재 → $WARM_CKPT"
  exit 1
fi
if [ ! -d "$JOINT_DIR" ] || [ -z "$(ls -A "$JOINT_DIR"/*.npz 2>/dev/null)" ]; then
  log "중단: joint_replay_dir 에 episode(.npz) 없음 → $JOINT_DIR"
  exit 1
fi

log "stage2-watchdog 가동. target=$TARGET_STEP warm=$WARM_CKPT joint_ratio=$JOINT_RATIO interval=${CHECK_INTERVAL}s"

while true; do
  if is_alive; then
    sleep "$CHECK_INTERVAL"
    continue
  fi
  log "프로세스 미검출(1차). ${CONFIRM}s 후 재확인."
  sleep "$CONFIRM"
  if is_alive; then
    log "재확인 결과 생존 → 오탐, 감시 계속."
    continue
  fi
  st="$(last_step)"
  log "프로세스 사망 확정. last_step=$st (target=$TARGET_STEP)"
  if [ "$st" -ge "$TARGET_STEP" ]; then
    log "목표 step 도달 → Stage2 완료로 간주. watchdog 종료."
    exit 0
  fi
  log "목표 미달 → 시작/crash resume."
  start_train
  sleep "$GRACE"
done
