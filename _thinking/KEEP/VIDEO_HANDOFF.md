# [핸드오프] 주행 동영상 저장 작업 — 새 세션용 (2026-05-23)

> ✅ **완료(2026-05-24).** scripts/record_drive.py 신규 작성·녹화 완료. mp4는 _thinking/KEEP/에
> 로컬 보존(*.mp4 gitignore라 git 제외): map_easy3_best_drive.mp4, oscher_best_drive.mp4.
> 사용자 확인: 2바퀴 잘 주행·기록 좋아 유지. 단 ★oscher는 --best(=lap-best 16.6초/policy 85step)로
> 추정 생성 — 사용자가 best로 지정한 eval-best(eval_return 336/policy 80step=step_80k.pt)와는 다름
> (lap_time 기준 vs eval_return 기준, 030 참조). 현 동영상 그대로 유지하기로 함.
> 아래는 작업 당시 핸드오프 원문(참고용).
>
> ---
> 이전 세션 컨텍스트가 길어져 동영상 작업을 새 세션으로 넘김. 아래만 보면 바로 착수 가능.
> 학습은 정지 상태(029 참조). 이 작업은 학습과 무관(시연/녹화만).

## 목표
map_easy3 best와 Oschersleben best 정책의 **주행을 각각 mp4로 녹화**해 `_thinking/KEEP/`에 저장.
- 산출물 예: `_thinking/KEEP/map_easy3_best_drive.mp4`, `_thinking/KEEP/oscher_best_drive.mp4`
- 번호증분/덮어쓰기 금지 정책 유의([[output-no-overwrite]]). (.gitignore에 *.mp4 등록됨 → git 제외, 로컬 보존)

## ★ 사용자 요구사항 (엄수)
- **맵이 다 보여야 한다** — 차 추적 카메라로 맵 일부만 나오면 안 됨. 트랙 전체가 프레임에 들어와야.
- **너무 축소해서 맵/시간(텍스트)이 조그맣게 보이면 안 된다** — 맵이 프레임을 적절히 꽉 채우되
  과도한 줌아웃 금지. lap time 텍스트도 읽을 수 있는 크기.
- 즉 카메라를 **맵 bounding box에 타이트하게 fit**(약간 여백)하는 게 핵심.

## 녹화 대상 명령 (정책 = --best 자동 선택)
```bash
cd ~/f1tenth_RL_project && source .venv/bin/activate
# map_easy3 최고(현재 6.1초)
python scripts/watch_drive.py --logdir runs/stage1_map_easy3 --task f1tenth_map_easy3 --best --episodes 3
# oschersleben 최고(현재 16.6초)
python scripts/watch_drive.py --logdir runs/stage2_oschersleben --task f1tenth_Oschersleben --best --episodes 3
```
watch_drive는 **화면 렌더만**(녹화 기능 없음) → 녹화하려면 아래 record 스크립트 신규 작성 필요.

## 환경 (확인 완료)
- `DISPLAY=:0` — WSLg X server 작동(GUI 창 뜸). 사용자가 watch_drive human mode 봤음.
- `imageio 2.35.1` + `imageio_ffmpeg` 설치됨 → mp4 인코딩 가능(`imageio.get_writer(path, fps=...)`).
- venv: /home/dlacksdn/f1tenth_RL_project/.venv (Python 3.8). ★ cwd 리셋 빈번 → 절대경로.

## 기술 세부 (조사 완료)
**렌더러**: `gym/f110_gym/envs/rendering.py`의 `EnvRenderer(pyglet.window.Window)`.
- 카메라는 **고정 orthographic**: `on_draw`(line ~290)에서 `glOrtho(self.left, self.right,
  self.bottom, self.top, 1, -1)`. **update_obs(line ~303)는 차 추적 안 함** — poses(차 폴리곤)와
  score_label만 갱신. 따라서 self.left/right/bottom/top만 맵 전체로 설정하면 맵 다 보임.
- `update_map`(grep으로 위치 확인)에서 map_points 생성 + 초기 카메라. zoom_level 기본 1.2.
- 좌표 스케일: 차 vertices가 `50. * get_vertices(...)` (1m≈50px). map_points도 동일 스케일 추정 →
  카메라 box도 그 스케일 기준(미터×50). map_points의 min/max로 box 잡고 여백 ~10% + 종횡비 보정.
- score_label: `'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'` 이미 화면에 표시됨(시간 OK).
- WINDOW_W/WINDOW_H, VIDEO_W/H(=600/400) 상수는 rendering.py / f110_env.py 상단 확인.

**프레임 캡처(pyglet GL buffer)**: render() 호출(=on_draw+flip) 직후,
```python
import pyglet, numpy as np
buf = pyglet.image.get_buffer_manager().get_color_buffer()
data = buf.get_image_data().get_data('RGB', buf.width * 3)
frame = np.frombuffer(data, dtype=np.uint8).reshape(buf.height, buf.width, 3)[::-1]  # GL은 bottom-up → flip
writer.append_data(frame)
```

**구현 권장**: `scripts/record_drive.py`를 watch_drive.py 복제 후 수정해 작성.
- `RenderDamy`(watch_drive.py:84) → `RecordDamy`로: step wrapped()에서 `f110.render(mode)` 후 위
  버퍼 캡처 → imageio writer.append_data. 에피소드 종료/episodes 소진 시 writer.close().
- 시작 시(첫 render 후) renderer.left/right/bottom/top을 맵 전체로 1회 설정(차 추적 비활성이라 고정 유지됨).
- mode는 `human_fast` 권장(빠른 녹화). fps는 env dt 기준(action_repeat=2, sim dt 확인) 또는 30 고정.
- ckpt 선택은 watch_drive의 --best/--ckpt 로직 그대로 재사용(이미 strict=False 패치됨).

## 함정/주의
- WSLg에서 pyglet 창이 실제로 떠야 buffer 캡처 가능(off-screen은 추가 설정 필요). DISPLAY=:0 확인됨.
- buffer가 RGBA일 수 있음 → 'RGB'로 강제 추출 or 4채널 받고 [:, :, :3].
- 카메라 fit 안 하면 맵이 화면 밖/일부만 → 사용자 요구 위반. **map_points min/max fit이 제1 검증 포인트.**
- best 파일은 partial(strict=False로 로드됨, 주행엔 critic 불필요). 정상.
- 산출물 검증: mp4 열어 ① 맵 전체 보이는지 ② 텍스트 읽히는지 ③ 차 주행/완주 보이는지.

## 참고 정책 경로
- map_easy3 best: runs/stage1_map_easy3/policy_best_lap6.1s_step224k.pt (full=step_224k.pt)
- oscher best: runs/stage2_oschersleben/policy_best_lap16.6s_step85k.pt (full=step_85k.pt)
- 백업: runs/stage2_oschersleben/KEEP/ (oscher 17.4s/16.6s + FULL)
- 관련: 029(정지/watch_drive --best), oscher_demo_policy_step80k.md(80k 백업 정책).
