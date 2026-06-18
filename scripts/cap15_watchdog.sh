#!/usr/bin/env bash
# cap15_watchdog.sh — Diffuser 데이터용 cap-15(V_MAX=15) Oschersleben 정책 학습 + crash 자동 resume.
#
# stage2_watchdog.sh 의 검증된 패턴 그대로. 차이는 (1) --v_max 15.0 (2) 전용 logdir
# (3) --steps 300000 (overnight 상한) (4) pgrep을 logdir로 식별(stage2와 task 동일).
# Dreamer 코어/차량 물리 무변경 — V_MAX는 backward-compat config 파라미터(기본 20.0).
# 학습 자체는 online이나 Diffuser offline 학습용 *데이터 수집*이라 과제 제약 무위반.
#
# 사용 (detached):
#   setsid nohup bash /home/dlacksdn/f1tenth_RL_project/scripts/cap15_watchdog.sh \
#     > /home/dlacksdn/f1tenth_RL_project/runs/cap15_oschersleben/watchdog.log 2>&1 < /dev/null &
set -u

PROJ=/home/dlacksdn/f1tenth_RL_project
VENDOR="$PROJ/vendor/dreamerv3-torch"

TASK=f1tenth_Oschersleben
LOGDIR="$PROJ/runs/cap15_oschersleben"
TARGET_STEP="${1:-300000}"
V_MAX=15.0

# stage2와 동일 warm-load 운영 파라미터(proven). v_max만 추가.
WARM_CKPT="$PROJ/runs/stage1_map_easy3/latest.pt"
JOINT_DIR="$PROJ/runs/stage1_map_easy3/train_eps"
WARM_LR_SCALE=0.5
JOINT_RATIO=0.3

METRICS="$LOGDIR/metrics.jsonl"
CHECK_INTERVAL=120
GRACE=90
CONFIRM=15

mkdir -p "$LOGDIR"
log(){ echo "[cap15-watchdog $(date '+%F %T')] $*"; }

# cap15 프로세스만 식별(logdir로 유일 — stage2와 task 동일하므로 logdir로 구분).
# ★ bracket-trick `[d]reamer`: pgrep -f 가 *그 패턴 문자열을 cmdline에 포함한 다른 셸 명령*
#   (모니터링/체크 명령 등)까지 매칭하는 cross-match 오탐 방지. 실제 python 프로세스
#   cmdline("python -u dreamer.py ... cap15_oschersleben")은 정상 매칭됨.
is_alive(){ pgrep -f "[d]reamer\.py.*cap15_oschersleben" >/dev/null 2>&1; }
# cap15 외 다른 dreamer 학습 생존 → GPU 점유 → OOM 위험.
other_dreamer(){ pgrep -af "[d]reamer\.py.*--envs 8" 2>/dev/null | grep -v "cap15_oschersleben"; }

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
  log "cap15 시작/resume (task=$TASK v_max=$V_MAX logdir=$LOGDIR)"
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

# --- 사전 안전 점검 (OOM/오타 fail-fast) ---
if other_dreamer >/dev/null; then
  log "중단: cap15 외 dreamer 학습 생존 → GPU 점유 중, OOM 위험. 종료."
  other_dreamer | while read -r l; do log "  ↳ $l"; done
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

log "cap15-watchdog 가동. target=$TARGET_STEP v_max=$V_MAX warm=$WARM_CKPT joint=$JOINT_RATIO interval=${CHECK_INTERVAL}s"

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
    log "목표 step 도달 → cap15 완료로 간주. watchdog 종료."
    exit 0
  fi
  log "목표 미달 → 시작/crash resume."
  start_train
  sleep "$GRACE"
done
