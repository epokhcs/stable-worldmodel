"""Export four representative episode videos from a Glitched Hue dataset.

Finds the longest successful and longest failed episode for each room
condition (blue+teleport, green+disabled) and writes four annotated MP4s:

    blue_success.mp4   — blue room, agent reached target
    blue_failure.mp4   — blue room, agent timed out
    green_success.mp4  — green room, agent reached target
    green_failure.mp4  — green room, agent timed out

Usage:
    python scripts/visualization/export_sample_videos.py
    python scripts/visualization/export_sample_videos.py --output-dir /workspace/videos
    python scripts/visualization/export_sample_videos.py --fps 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from stable_worldmodel.data.utils import get_cache_dir

# Reuse room-classification constants from find_glitched_hue_episodes
_BLUE_REF = np.array([0, 0, 255], dtype=np.float32)
_GREEN_REF = np.array([0, 180, 0], dtype=np.float32)
_PATCH = (slice(20, 40), slice(20, 40))


def _classify_room(frame: np.ndarray) -> str:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    patch = arr[_PATCH[0], _PATCH[1], :3]
    mean_rgb = patch.mean(axis=(0, 1))
    return 'blue' if np.linalg.norm(mean_rgb - _BLUE_REF) <= np.linalg.norm(mean_rgb - _GREEN_REF) else 'green'


def _find_candidates(h5_path: Path) -> dict[str, int]:
    """Return episode indices for the four target conditions.

    Strategy: pick the *shortest* matching episode so the video is concise.
    Falls back to the first matching episode if there is only one.
    """
    with h5py.File(h5_path, 'r') as f:
        ep_len = f['ep_len'][:].astype(int)
        ep_offset = f['ep_offset'][:].astype(int)
        terminated = f['terminated'][:].astype(bool)
        pixels = f['pixels']

        n = len(ep_len)
        success = np.zeros(n, dtype=bool)
        room = np.empty(n, dtype=object)

        for i, (start, length) in enumerate(zip(ep_offset, ep_len)):
            success[i] = terminated[int(start):int(start + length)].any()
            room[i] = _classify_room(pixels[int(start)])

    blue = room == 'blue'
    green = room == 'green'

    def _pick(mask: np.ndarray) -> int:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return -1
        # longest episode — agent and target are far apart, more interesting to watch
        return int(indices[np.argmax(ep_len[indices])])

    return {
        'blue_success':   _pick(blue & success),
        'blue_failure':   _pick(blue & ~success),
        'green_success':  _pick(green & success),
        'green_failure':  _pick(green & ~success),
    }


def export_sample_videos(
    h5_path: Path,
    output_dir: Path,
    fps: int = 10,
) -> dict[str, Path]:
    """Export four sample videos and return a mapping of label → output path."""
    from scripts.visualization.episode_to_mp4 import episode_to_mp4

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _find_candidates(h5_path)

    results: dict[str, Path] = {}
    for label, ep_idx in candidates.items():
        if ep_idx == -1:
            print(f'  [skip] {label}: no matching episode found')
            continue
        out = output_dir / f'{label}.mp4'
        print(f'  Exporting episode {ep_idx:>5d} → {out.name} ...')
        episode_to_mp4(h5_path=h5_path, episode_idx=ep_idx, output_path=out, fps=fps)
        results[label] = out
        print(f'  Done: {out}')

    return results


def parse_args() -> argparse.Namespace:
    default_h5 = Path(get_cache_dir()) / 'glitched_hue_tworoom.h5'
    default_out = Path(get_cache_dir()) / 'videos'
    parser = argparse.ArgumentParser(
        description='Export blue/green success/failure sample videos from a Glitched Hue dataset.'
    )
    parser.add_argument(
        '--h5-path', type=Path, default=default_h5,
        help=f'HDF5 dataset path (default: {default_h5})',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=default_out,
        help=f'Directory to write MP4 files (default: {default_out})',
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frames per second (default: 10)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'Dataset not found: {h5_path}')

    print(f'Dataset : {h5_path}')
    print(f'Output  : {args.output_dir}')
    results = export_sample_videos(h5_path, args.output_dir.expanduser().resolve(), fps=args.fps)
    print(f'\nExported {len(results)}/4 videos:')
    for label, path in results.items():
        print(f'  {label:<16} → {path}')


if __name__ == '__main__':
    main()
