#!/usr/bin/env python3
"""
Phase 1-1: Measure GapFollower baseline lap times.
5 ep × 2 maps, eval pose fixed, no render.
Output: _thinking/notes/gap_follower_baseline.md
"""
import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'gym'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'pkg', 'src'))

from f110_gym.envs.f110_env import F110Env
from pkg.drivers import GapFollower

MAPS_SRC = os.path.join(PROJECT_ROOT, 'pkg', 'src', 'pkg', 'maps')
MAPS_OUT = os.path.join(PROJECT_ROOT, 'maps')
NOTES_DIR = os.path.join(PROJECT_ROOT, '_thinking', 'notes')

# Fixed start poses [x, y, theta] (v3 결정 #28: eval pose 고정)
# map_easy3: dqn.py 포즈는 reset 시 wall collision 발생 → centerline idx=0 사용
# Oschersleben: main.py에서 확인된 포즈
START_POSES = {
    'map_easy3':    np.array([8.620, 11.860, 2.356]),
    'Oschersleben': np.array([0.0702245, 0.3002981, 2.79787]),
}
# A11/A13 fallback thresholds (결정 #28)
FALLBACK = {
    'map_easy3':    {'median': 45.0, 'min': 45.0},
    'Oschersleben': {'median': 110.0, 'min': 110.0},
}
SIM_TIME_LIMIT = 300.0  # seconds per episode (sim time)
N_EPISODES = 5


# ---------------------------------------------------------------------------
def get_start_pose(map_name):
    if map_name in START_POSES:
        return START_POSES[map_name].copy()
    # Derive from centerline CSV first point
    csv = os.path.join(MAPS_OUT, f'{map_name}_centerline.csv')
    if os.path.exists(csv):
        data = np.loadtxt(csv, delimiter=',', skiprows=1)
        x0, y0, tx0, ty0 = data[0, 1], data[0, 2], data[0, 3], data[0, 4]
        theta0 = float(np.arctan2(ty0, tx0))
        print(f"  start from centerline: x={x0:.3f} y={y0:.3f} θ={theta0:.3f}")
        return np.array([x0, y0, theta0])
    raise RuntimeError(
        f"No start pose for '{map_name}'. "
        f"Run extract_centerline.py first, or add to START_POSES, or use --fallback."
    )


def run_episode(env, start_pose, time_limit):
    """Run one GapFollower episode; return first lap_time or None on DNF."""
    poses = start_pose[np.newaxis, :]          # (1, 3)
    obs, _, _, _ = env.reset(poses)
    # dqn.py pattern: ignore done from reset (start pose near wall can
    # trigger a false collision in reset's internal zero-action step).
    done = False

    gf = GapFollower()
    step = 0
    prev_lap_count = 0

    while not done and step * env.timestep < time_limit:
        speed, steer = gf.process_lidar(obs['scans'][0])
        action = np.array([[steer, speed]])     # shape (1, 2): [steer, speed]
        obs, _, done, _ = env.step(action)
        step += 1

        # collision → DNF
        if obs['collisions'][0] > 0:
            return None

        # first lap complete
        if int(obs['lap_counts'][0]) > prev_lap_count:
            return float(obs['lap_times'][0])

    return None  # timeout


def measure_map(map_name, n_episodes, time_limit):
    map_path = os.path.join(MAPS_SRC, map_name)   # gym appends .yaml
    start_pose = get_start_pose(map_name)

    env = F110Env(map=map_path, map_ext='.png', num_agents=1,
                  timestep=0.01, ego_idx=0)

    lap_times, dnf_count = [], 0
    wall_t0 = time.time()

    for ep in range(n_episodes):
        t = run_episode(env, start_pose, time_limit)
        if t is not None:
            lap_times.append(t)
            print(f"  ep {ep+1}/{n_episodes}: {t:.2f} s")
        else:
            dnf_count += 1
            print(f"  ep {ep+1}/{n_episodes}: DNF")

    env.close()
    wall_elapsed = time.time() - wall_t0
    return lap_times, dnf_count, wall_elapsed


# ---------------------------------------------------------------------------
def save_md(results, n_episodes):
    os.makedirs(NOTES_DIR, exist_ok=True)
    path = os.path.join(NOTES_DIR, 'gap_follower_baseline.md')
    lines = [
        '# GapFollower Baseline (Phase 1-1)',
        '',
        '> scripts/measure_gap_follower.py 출력.',
        f'> 에피소드: {n_episodes}회 × 맵',
        '',
        '| Map | Median (s) | Min (s) | n | DNF | Fallback |',
        '|---|---|---|---|---|---|',
    ]
    for m, r in results.items():
        med = f"{r['median']:.2f}" if r['median'] is not None else '-'
        mn  = f"{r['min']:.2f}"    if r['min']    is not None else '-'
        n   = len(r['lap_times'])
        fb  = '✅' if r.get('fallback') else '❌'
        lines.append(f'| {m} | {med} | {mn} | {n} | {r["dnf"]} | {fb} |')

    lines += ['', '## A11/A13 DreamerV3 통과 기준 (Phase 1-1 확정값)', '']
    for m, r in results.items():
        if r['median'] is not None:
            bl = r['median']
            if m == 'map_easy3':
                lines.append(f'- **A11 {m}**: DreamerV3 median ≤ {bl * 1.5:.1f} s  (= baseline {bl:.2f} s × 1.5)')
            else:
                lines.append(f'- **A13 {m}**: DreamerV3 median ≤ 120 s, best ≤ 110 s  (GapFollower baseline {bl:.2f} s)')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved → {path}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', nargs='+', default=['map_easy3', 'Oschersleben'])
    parser.add_argument('--episodes', type=int, default=N_EPISODES)
    parser.add_argument('--time-limit', type=float, default=SIM_TIME_LIMIT,
                        help='Max sim seconds per episode (default 300)')
    parser.add_argument('--fallback', action='store_true',
                        help='Skip measurement; adopt A11=45s / A13=110s fallback values')
    args = parser.parse_args()

    results = {}

    if args.fallback:
        print("--fallback: using hardcoded A11/A13 values")
        for m in args.maps:
            fb = FALLBACK.get(m, {'median': None, 'min': None})
            results[m] = {'lap_times': [], 'dnf': args.episodes,
                          'fallback': True, **fb}
            print(f"  {m}: median={fb['median']} s  min={fb['min']} s")
    else:
        for m in args.maps:
            print(f'\n=== {m} ===')
            try:
                lts, dnf, wall = measure_map(m, args.episodes, args.time_limit)
                median = float(np.median(lts)) if lts else None
                min_t  = float(np.min(lts))    if lts else None
                results[m] = {'lap_times': lts, 'dnf': dnf, 'fallback': False,
                               'median': median, 'min': min_t}
                print(f"  → n={len(lts)} median={median} min={min_t} DNF={dnf}  wall={wall:.0f}s")
            except Exception as exc:
                print(f"  ERROR: {exc}")
                print(f"  → fallback for {m}")
                fb = FALLBACK.get(m, {'median': None, 'min': None})
                results[m] = {'lap_times': [], 'dnf': args.episodes,
                               'fallback': True, **fb}

    save_md(results, args.episodes)
    print('\nDone.')


if __name__ == '__main__':
    main()
