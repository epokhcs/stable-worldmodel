"""Compute and save episode-level statistics for a Glitched Hue dataset.

Reads the collected HDF5 file and writes a Markdown report next to it with:
  - Overall success rate
  - Per-condition breakdown (blue+teleport vs green+no-teleport)
  - Steps-to-success statistics (min / mean / max)
  - Teleport pixel usage counts

Usage:
    python scripts/data/compute_episode_stats.py
    python scripts/data/compute_episode_stats.py --h5-path /path/to/file.h5
    python scripts/data/compute_episode_stats.py --output /path/to/episode_stats.md
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc decompressor
import numpy as np

from stable_worldmodel.data.utils import get_cache_dir

# Reference colours for room classification (must match collect_glitched_hue.py)
_BLUE_REF = np.array([0, 0, 255], dtype=np.float32)
_GREEN_REF = np.array([0, 180, 0], dtype=np.float32)
_PATCH = (slice(20, 40), slice(20, 40))


def _classify_room(frame: np.ndarray) -> str:
    """Return 'blue' or 'green' by comparing a top-left pixel patch to reference colours."""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    patch = arr[_PATCH[0], _PATCH[1], :3]
    mean_rgb = patch.mean(axis=(0, 1))
    dist_blue = float(np.linalg.norm(mean_rgb - _BLUE_REF))
    dist_green = float(np.linalg.norm(mean_rgb - _GREEN_REF))
    return 'blue' if dist_blue <= dist_green else 'green'


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return 'N/A'
    return f'{100.0 * num / denom:.1f}%'


def _steps_stats(ep_len: np.ndarray, mask: np.ndarray) -> str:
    """Return 'min / mean / max' string for episodes matching mask, or 'N/A'."""
    vals = ep_len[mask]
    if vals.size == 0:
        return 'N/A'
    return f'{int(vals.min())} / {vals.mean():.1f} / {int(vals.max())}'


def compute_stats(h5_path: Path) -> dict:
    """Return a dict of statistics computed from the HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        required = {'ep_len', 'ep_offset', 'terminated', 'teleported', 'pixels'}
        missing = sorted(required - set(f.keys()))
        if missing:
            raise KeyError(f'Dataset missing required keys: {", ".join(missing)}')

        ep_len = f['ep_len'][:].astype(int)
        ep_offset = f['ep_offset'][:].astype(int)
        terminated = f['terminated'][:].astype(bool)
        teleported_steps = f['teleported'][:].astype(bool)
        pixels = f['pixels']  # keep lazy — read one frame per episode

        n_episodes = len(ep_len)
        success = np.zeros(n_episodes, dtype=bool)
        ep_teleport_count = np.zeros(n_episodes, dtype=int)
        room = np.empty(n_episodes, dtype=object)

        for i, (start, length) in enumerate(zip(ep_offset, ep_len)):
            sl = slice(int(start), int(start + length))
            success[i] = bool(terminated[sl].any())
            ep_teleport_count[i] = int(teleported_steps[sl].sum())
            room[i] = _classify_room(pixels[int(start)])

    blue = room == 'blue'
    green = room == 'green'
    used_teleport = ep_teleport_count > 0

    return {
        'n_episodes': n_episodes,
        'n_success': int(success.sum()),
        'ep_len': ep_len,
        'success': success,
        'blue': blue,
        'green': green,
        'used_teleport': used_teleport,
        'ep_teleport_count': ep_teleport_count,
        'teleported_steps_total': int(teleported_steps.sum()),
    }


def build_report(stats: dict, h5_path: Path) -> str:
    n = stats['n_episodes']
    n_ok = stats['n_success']
    ep_len = stats['ep_len']
    success = stats['success']
    blue = stats['blue']
    green = stats['green']
    used_tp = stats['used_teleport']
    ep_tp = stats['ep_teleport_count']

    # Per-condition masks
    b_ok = blue & success
    g_ok = green & success
    b_tp = blue & used_tp
    g_tp = green & used_tp

    timestamp = dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')

    lines = [
        f'# Episode Statistics — {h5_path.stem}',
        '',
        f'Generated: {timestamp}  ',
        f'Dataset: `{h5_path}`',
        '',
        '---',
        '',
        '## Overall',
        '',
        '| Metric | Value |',
        '|---|---|',
        f'| Total episodes | {n:,} |',
        f'| Successful (reached target) | {n_ok:,} ({_pct(n_ok, n)}) |',
        f'| Failed / timed-out | {n - n_ok:,} ({_pct(n - n_ok, n)}) |',
        f'| Steps to success (min / mean / max) | {_steps_stats(ep_len, success)} |',
        '',
        '---',
        '',
        '## By Condition',
        '',
        '| Condition | Total | Successful | Success rate | Steps to success (min / mean / max) |',
        '|---|---:|---:|---:|---|',
        f'| Blue room + teleport enabled | {int(blue.sum()):,} | {int(b_ok.sum()):,} | {_pct(int(b_ok.sum()), int(blue.sum()))} | {_steps_stats(ep_len, b_ok)} |',
        f'| Green room + teleport disabled | {int(green.sum()):,} | {int(g_ok.sum()):,} | {_pct(int(g_ok.sum()), int(green.sum()))} | {_steps_stats(ep_len, g_ok)} |',
        '',
        '---',
        '',
        '## Teleport Usage',
        '',
        '| Metric | Count |',
        '|---|---:|',
        f'| Total teleport-step events | {stats["teleported_steps_total"]:,} |',
        f'| Episodes using teleport (≥1 step) | {int(used_tp.sum()):,} ({_pct(int(used_tp.sum()), n)}) |',
        f'| — in blue room | {int(b_tp.sum()):,} |',
        f'| — in green room | {int(g_tp.sum()):,} |',
        f'| Mean teleport steps per episode (blue) | {ep_tp[blue].mean():.2f} |',
        f'| Mean teleport steps per episode (green) | {ep_tp[green].mean():.2f} |',
        '',
    ]
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    default_h5 = Path(get_cache_dir()) / 'glitched_hue_tworoom.h5'
    parser = argparse.ArgumentParser(
        description='Compute episode statistics for a Glitched Hue HDF5 dataset.'
    )
    parser.add_argument(
        '--h5-path',
        type=Path,
        default=default_h5,
        help=f'Path to the HDF5 dataset (default: {default_h5})',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output .md path (default: episode_stats.md in same directory as the .h5)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'Dataset not found: {h5_path}')

    output_path = args.output or (h5_path.parent / 'episode_stats.md')
    output_path = output_path.expanduser().resolve()

    print(f'Reading {h5_path} ...')
    stats = compute_stats(h5_path)
    report = build_report(stats, h5_path)

    output_path.write_text(report, encoding='utf-8')
    print(f'Stats written to {output_path}')

    # Print a brief summary to stdout too
    n = stats['n_episodes']
    n_ok = stats['n_success']
    print(f'  Total episodes  : {n:,}')
    print(f'  Successful       : {n_ok:,} ({_pct(n_ok, n)})')
    print(f'  Blue episodes    : {int(stats["blue"].sum()):,}')
    print(f'  Green episodes   : {int(stats["green"].sum()):,}')
    print(f'  Teleport events  : {stats["teleported_steps_total"]:,} steps across {int(stats["used_teleport"].sum()):,} episodes')


if __name__ == '__main__':
    main()
