#!/usr/bin/env python3
"""
Phase 1-1: Extract track centerline via skimage skeletonize.
Output: maps/{map_name}_centerline.csv  (cols: s, x, y, tx, ty)
        _thinking/notes/track_length.md
"""
import argparse
import os
import sys

import numpy as np
import yaml
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MAPS_SRC = os.path.join(PROJECT_ROOT, 'pkg', 'src', 'pkg', 'maps')
MAPS_OUT = os.path.join(PROJECT_ROOT, 'maps')
NOTES_DIR = os.path.join(PROJECT_ROOT, '_thinking', 'notes')

PRIOR_ESTIMATES = {'map_easy3': 70.0, 'Oschersleben': 300.0}

# Start pose (world x, y) ON the drivable track — used to select the track-ribbon
# free CC (implementation/008). Must match dreamer_f1tenth TRACK_CONFIGS default_pose.
# (x, y, yaw): yaw = 주행 시작 heading. centerline +s(arclength 증가 방향)를 이 heading에
# 맞춰 정렬한다 → 전진 주행 시 progress(arclen_delta)>0 보장 (Phase 4 reward 정합).
# 값은 dreamer_f1tenth TRACK_CONFIGS default_pose와 일치해야 함.
START_POSES = {
    # map_easy3: green-ribbon on-track pose (implementation/008). 옛 (8.620,11.860)은
    # 트랙 밖(outer free CC)이라 잘못된 centerline을 냈음 — max-clearance on-track으로 교체.
    'map_easy3': (1.02, -14.66, -2.819842),
    'Oschersleben': (0.0702245, 0.3002981, 2.79787),
}


# ---------------------------------------------------------------------------
def load_free_mask(map_name):
    yaml_path = os.path.join(MAPS_SRC, map_name + '.yaml')
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = float(meta['resolution'])
    origin = meta['origin']           # [ox, oy, theta_or_0]
    negate = int(meta.get('negate', 0))
    free_thresh = float(meta.get('free_thresh', 0.196))

    img_file = meta.get('image', map_name + '.png')
    img = Image.open(os.path.join(MAPS_SRC, img_file)).convert('L')
    arr = np.array(img, dtype=np.float32)

    if negate:
        arr = 255.0 - arr

    # ROS convention: occ_prob = 1 - pixel/255; free if occ_prob < free_thresh
    free_mask = arr > (1.0 - free_thresh) * 255.0
    H, W = free_mask.shape
    return free_mask, resolution, origin, H, W


def px_to_world(rows, cols, resolution, origin, H):
    """Image pixel (row, col) → world (x, y). ROS: origin is bottom-left corner."""
    ox, oy = float(origin[0]), float(origin[1])
    x = ox + cols * resolution
    y = oy + (H - 1 - rows) * resolution
    return x, y


def world_to_px(x, y, resolution, origin, H):
    """World (x, y) → image pixel (row, col). Inverse of px_to_world."""
    ox, oy = float(origin[0]), float(origin[1])
    col = int(round((x - ox) / resolution))
    row = int((H - 1) - round((y - oy) / resolution))
    return row, col


# ---------------------------------------------------------------------------
def keep_largest_cc(skel):
    labeled = label(skel, connectivity=2)
    if labeled.max() == 0:
        return skel
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def prune_branches(skel, iterations=5):
    """Iteratively remove degree-1 skeleton pixels to eliminate dead-end branches."""
    s = skel.copy()
    for _ in range(iterations):
        rows, cols = np.where(s)
        to_remove = []
        for r, c in zip(rows, cols):
            nbr_count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr, dc) == (0, 0):
                        continue
                    if 0 <= r+dr < s.shape[0] and 0 <= c+dc < s.shape[1]:
                        if s[r+dr, c+dc]:
                            nbr_count += 1
            if nbr_count == 1:
                to_remove.append((r, c))
        if not to_remove:
            break
        for r, c in to_remove:
            s[r, c] = False
    return s


def order_skeleton_loop(skel_pts):
    """Walk skeleton into an ordered sequence using direction-continuation greedy."""
    pt_set = {(int(r), int(c)) for r, c in skel_pts}

    def nbrs(r, c):
        return [(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r+dr, c+dc) in pt_set]

    # Start from the pixel with fewest neighbours (a cleaner straight section)
    min_deg, start = 9, None
    for p in pt_set:
        d = len(nbrs(*p))
        if d < min_deg:
            min_deg, start = d, p

    path = [start]
    visited = {start}
    cur = start
    prev_dir = (0, 0)

    while True:
        candidates = [n for n in nbrs(*cur) if n not in visited]
        if not candidates:
            break
        if len(candidates) > 1 and prev_dir != (0, 0):
            # prefer direction closest to previous heading
            def align(n):
                dr, dc = n[0] - cur[0], n[1] - cur[1]
                return -(dr * prev_dir[0] + dc * prev_dir[1])
            candidates.sort(key=align)
        nxt = candidates[0]
        prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
        path.append(nxt)
        visited.add(nxt)
        cur = nxt

    return np.array(path, dtype=float)


# ---------------------------------------------------------------------------
def extract_centerline(map_name, verify=False):
    free_mask, resolution, origin, H, W = load_free_mask(map_name)
    free_frac = 100.0 * free_mask.sum() / free_mask.size
    print(f"[{map_name}] size={W}x{H} res={resolution} m/px  free={free_mask.sum()} ({free_frac:.1f}%)")

    # ★ The maps draw walls as THIN lines on a ~99% free background, so free space
    # splits into outer / track-ribbon / infield CCs (4-connectivity). Skeletonizing
    # ALL free + keep_largest_cc selects the OUTER region → wrong loop (the original
    # Phase 1-1 bug, implementation/008). Instead isolate the free CC that contains
    # the start pose (= the drivable track ribbon) and skeletonize only that.
    if map_name in START_POSES:
        sx, sy, _syaw = START_POSES[map_name]
        srow, scol = world_to_px(sx, sy, resolution, origin, H)
        labeled = label(free_mask, connectivity=1)  # 4-conn: thin walls block leaks
        start_label = labeled[srow, scol]
        if start_label == 0:
            raise RuntimeError(
                f"{map_name}: start pose ({sx},{sy})→px({srow},{scol}) is NOT free "
                f"(label 0). Check START_POSES / map registration."
            )
        ribbon = labeled == start_label
        rib_frac = 100.0 * ribbon.sum() / ribbon.size
        print(f"  ribbon CC (start-pose) = {int(ribbon.sum())} px ({rib_frac:.1f}%)")
        skel = skeletonize(ribbon)
    else:
        # No start pose known — legacy fallback (largest free skeleton CC).
        skel = keep_largest_cc(skeletonize(free_mask))
    skel = prune_branches(skel, iterations=10)
    rows, cols = np.where(skel)
    n_pts = len(rows)
    print(f"  skeleton: {n_pts} px")

    if n_pts == 0:
        raise RuntimeError(f"Empty skeleton for {map_name} — check threshold")

    ordered = order_skeleton_loop(list(zip(rows.tolist(), cols.tolist())))
    r_o, c_o = ordered[:, 0], ordered[:, 1]
    x, y = px_to_world(r_o, c_o, resolution, origin, H)

    # Orient the loop so +s (increasing index) matches the start heading, so that
    # forward driving yields progress (arclen_delta) > 0 in Phase 4 reward. The
    # skeleton walk direction is otherwise arbitrary. (implementation/008)
    if map_name in START_POSES:
        sx, sy, syaw = START_POSES[map_name]
        ci = int(np.argmin((x - sx) ** 2 + (y - sy) ** 2))
        nxt = (ci + 1) % len(x)
        if np.cos(syaw) * (x[nxt] - x[ci]) + np.sin(syaw) * (y[nxt] - y[ci]) < 0:
            x = x[::-1].copy()
            y = y[::-1].copy()
            print("  reversed centerline orientation to match start heading (+s = forward)")

    # arclength
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dx[0] = dy[0] = 0.0
    s = np.cumsum(np.sqrt(dx**2 + dy**2))
    s[0] = 0.0
    L_track = float(s[-1])

    # tangent (unit, via central differences over arc-length param)
    tx = np.gradient(x, s)
    ty = np.gradient(y, s)
    norm = np.maximum(np.sqrt(tx**2 + ty**2), 1e-9)
    tx /= norm
    ty /= norm

    # save CSV
    os.makedirs(MAPS_OUT, exist_ok=True)
    csv_path = os.path.join(MAPS_OUT, f'{map_name}_centerline.csv')
    np.savetxt(csv_path, np.column_stack([s, x, y, tx, ty]),
               delimiter=',', header='s,x,y,tx,ty', comments='', fmt='%.6f')
    print(f"  saved {csv_path}  ({len(s)} pts)  L_track={L_track:.2f} m")

    if verify and map_name in PRIOR_ESTIMATES:
        est = PRIOR_ESTIMATES[map_name]
        ratio = L_track / est
        flag = 'OK' if 0.7 <= ratio <= 1.3 else 'WARNING: >30% from prior estimate'
        print(f"  prior={est:.0f}m  ratio={ratio:.2f}  [{flag}]")

    return L_track, csv_path


def save_track_length_md(results):
    os.makedirs(NOTES_DIR, exist_ok=True)
    path = os.path.join(NOTES_DIR, 'track_length.md')
    lines = [
        '# Track Length (Phase 1-1 측정)',
        '',
        '> scripts/extract_centerline.py 출력.',
        '',
        '| Map | L_track (m) | 추정 (m) | 비율 |',
        '|---|---|---|---|',
    ]
    for name, L in results.items():
        est = PRIOR_ESTIMATES.get(name)
        est_s = f'{est:.0f}' if est else '-'
        ratio_s = f'{L / est:.2f}' if est else '-'
        lines.append(f'| {name} | {L:.2f} | {est_s} | {ratio_s} |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nTrack lengths → {path}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract track centerline')
    parser.add_argument('--maps', nargs='+', default=['map_easy3', 'Oschersleben'])
    parser.add_argument('--verify', action='store_true',
                        help='Compare L_track against prior estimates (±30% check)')
    args = parser.parse_args()

    results = {}
    for m in args.maps:
        print(f'\n=== {m} ===')
        L, _ = extract_centerline(m, verify=args.verify)
        results[m] = L

    save_track_length_md(results)
    print('\nDone.')
