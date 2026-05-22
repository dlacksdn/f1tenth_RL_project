#!/usr/bin/env bash
# train_watchdog.sh — Phase 5 학습 crash 자동 재개 감시 (envs=8 멀티프로세스 무인 안정성)
#
# 동작:
#   - CHECK_INTERVAL마다 dreamer.py 메인 프로세스 생존 확인.
#   - 죽었으면 metrics.jsonl의 last_step을 읽어 판별:
#       * last_step >= TARGET_STEP  → 정상 완료로 간주, watchdog 종료(재시작 안 함).
#       * 그 외                      → crash 판단, 같은 logdir로 resume(latest.pt + train_eps).
#   - ★ 실행 중인 학습은 절대 죽이거나 재시작하지 않는다(생존 시 감시만).
#
# 사용 (detached):
#   cd /home/dlacksdn/f1tenth_RL_project
#   setsid nohup ./scripts/train_watchdog.sh f1tenth_map_easy3 \
#     runs/stage1_map_easy3 500000 \
#     > runs/stage1_map_easy3/watchdog.log 2>&1 < /dev/null &
#
# 인자 (모두 선택, 기본=Stage 1):
#   $1 TASK         (default f1tenth_map_easy3)
#   $2 LOGDIR_REL   (default runs/stage1_map_easy3, PROJ 기준 상대 또는 절대)
#   $3 TARGET_STEP  (default 500000  = configs steps=5e5, env-step 기준)
set -u

PROJ=/home/dlacksdn/f1tenth_RL_project
VENDOR="$PROJ/vendor/dreamerv3-torch"

TASK="${1:-f1tenth_map_easy3}"
LOGDIR_IN="${2:-runs/stage1_map_easy3}"
TARGET_STEP="${3:-500000}"

# LOGDIR 절대경로화
case "$LOGDIR_IN" in
  /*) LOGDIR="$LOGDIR_IN" ;;
  *)  LOGDIR="$PROJ/$LOGDIR_IN" ;;
esac
METRICS="$LOGDIR/metrics.jsonl"

CHECK_INTERVAL=120   # 생존 체크 주기(초). 체크포인트가 ~15분마다이므로 2분이면 충분.
GRACE=90             # resume 후 프로세스 안정화 대기(초).
CONFIRM=15           # "프로세스 없음" 1차 감지 후 재확인 대기(순간 측정 오탐 방지).

log(){ echo "[watchdog $(date '+%F %T')] $*"; }

# 메인 학습 프로세스 생존 여부 (envs=8 + 해당 task 매칭)
is_alive(){ pgrep -f "dreamer.py.*--task ${TASK}.*--envs 8" >/dev/null 2>&1; }

# metrics.jsonl에서 최대 step 추출 (없으면 0)
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
  log "resume 시작 (task=$TASK, logdir=$LOGDIR)"
  cd "$VENDOR" || { log "cd vendor 실패"; return 1; }
  # shellcheck disable=SC1091
  source "$PROJ/.venv/bin/activate"
  setsid nohup python -u dreamer.py --configs f1tenth \
    --task "$TASK" --logdir "$LOGDIR" \
    --envs 8 --parallel True --log_every 500 \
    >> "$LOGDIR/train.log" 2>&1 < /dev/null &
  local pid=$!
  log "재시작 발사 PID=$pid"
}

log "watchdog 가동. task=$TASK logdir=$LOGDIR target_step=$TARGET_STEP interval=${CHECK_INTERVAL}s"
log "메인 생존 시 감시만 수행. 학습을 죽이거나 재시작하지 않음."

while true; do
  if is_alive; then
    sleep "$CHECK_INTERVAL"
    continue
  fi

  # 1차 감지 → CONFIRM 후 재확인(짧은 순간 미검출 오탐 방지)
  log "프로세스 미검출(1차). ${CONFIRM}s 후 재확인."
  sleep "$CONFIRM"
  if is_alive; then
    log "재확인 결과 생존 → 오탐, 감시 계속."
    continue
  fi

  st="$(last_step)"
  log "프로세스 사망 확정. last_step=$st (target=$TARGET_STEP)"
  if [ "$st" -ge "$TARGET_STEP" ]; then
    log "목표 step 도달 → 학습 완료로 간주. watchdog 종료."
    exit 0
  fi

  log "목표 미달 → crash 판단. 같은 logdir로 resume."
  start_train
  sleep "$GRACE"
done
