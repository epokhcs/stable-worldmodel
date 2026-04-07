from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from stable_worldmodel.data.utils import get_cache_dir

BLUE_REF = np.array([0, 0, 255], dtype=np.float32)
GREEN_REF = np.array([0, 180, 0], dtype=np.float32)
PATCH_SLICE = (slice(20, 40), slice(20, 40))


def _to_hwc(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f'Expected a 3D image array, got {arr.shape}')
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def classify_room(frame: np.ndarray) -> tuple[str, np.ndarray]:
    hwc = _to_hwc(frame)
    patch = hwc[PATCH_SLICE[0], PATCH_SLICE[1], :3].astype(np.float32)
    mean_rgb = patch.mean(axis=(0, 1))
    dist_blue = np.linalg.norm(mean_rgb - BLUE_REF)
    dist_green = np.linalg.norm(mean_rgb - GREEN_REF)
    room = 'blue' if dist_blue <= dist_green else 'green'
    return room, mean_rgb


def format_row(ep_idx: int, ep_len: int, room: str, teleported: bool) -> str:
    behavior = 'teleport' if teleported else 'door'
    return (
        f'ep={ep_idx:5d} | len={ep_len:3d} | room={room:<5s} '
        f'| behavior={behavior}'
    )


def parse_args() -> argparse.Namespace:
    default_h5 = Path(get_cache_dir()) / 'glitched_hue_tworoom.h5'
    parser = argparse.ArgumentParser(
        description='Find glitched-hue episodes by color, teleport use, and length.'
    )
    parser.add_argument(
        '--h5',
        type=Path,
        default=default_h5,
        help=f'Path to the HDF5 dataset (default: {default_h5}).',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of episodes to print after sorting by descending length.',
    )
    parser.add_argument(
        '--room',
        choices=['any', 'blue', 'green'],
        default='any',
        help='Filter by room color.',
    )
    parser.add_argument(
        '--behavior',
        choices=['any', 'teleport', 'door'],
        default='any',
        help='Filter by whether the episode used the teleport pixel or main door.',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only print summary counts, not the top episode list.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = args.h5.expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'Dataset not found: {h5_path}')

    with h5py.File(h5_path, 'r') as f:
        required = {'pixels', 'ep_len', 'ep_offset', 'teleported'}
        missing = sorted(required - set(f.keys()))
        if missing:
            raise KeyError(
                f'Dataset is missing required keys: {", ".join(missing)}'
            )

        ep_len = f['ep_len'][:].astype(int)
        ep_offset = f['ep_offset'][:].astype(int)
        teleported_steps = f['teleported'][:].astype(bool)
        pixels = f['pixels']

        rooms = np.empty(len(ep_len), dtype=object)
        teleported_ep = np.zeros(len(ep_len), dtype=bool)

        for ep_idx, (start, length) in enumerate(zip(ep_offset, ep_len)):
            room, _ = classify_room(pixels[int(start)])
            rooms[ep_idx] = room
            teleported_ep[ep_idx] = teleported_steps[
                int(start) : int(start + length)
            ].any()

    blue_count = int(np.sum(rooms == 'blue'))
    green_count = int(np.sum(rooms == 'green'))
    teleport_count = int(np.sum(teleported_ep))
    door_count = int(np.sum(~teleported_ep))

    print(f'Dataset: {h5_path}')
    print(f'Total episodes: {len(ep_len)}')
    print(f'Blue room episodes: {blue_count}')
    print(f'Green room episodes: {green_count}')
    print(f'Teleport episodes: {teleport_count}')
    print(f'Main-door / no-teleport episodes: {door_count}')

    if args.summary_only:
        return

    mask = np.ones(len(ep_len), dtype=bool)
    if args.room != 'any':
        mask &= rooms == args.room
    if args.behavior == 'teleport':
        mask &= teleported_ep
    elif args.behavior == 'door':
        mask &= ~teleported_ep

    matching = np.flatnonzero(mask)
    if matching.size == 0:
        print('\nNo episodes matched the requested filters.')
        return

    order = np.argsort(ep_len[matching])[::-1]
    top_idx = matching[order[: args.top_k]]

    print(
        '\nTop matching episodes '
        f'(room={args.room}, behavior={args.behavior}, top_k={args.top_k}):'
    )
    for ep_idx in top_idx:
        print(
            format_row(
                ep_idx=int(ep_idx),
                ep_len=int(ep_len[ep_idx]),
                room=str(rooms[ep_idx]),
                teleported=bool(teleported_ep[ep_idx]),
            )
        )


if __name__ == '__main__':
    main()
