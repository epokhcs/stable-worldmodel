from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from stable_worldmodel.data.utils import get_cache_dir


def _to_hwc(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to HWC uint8 RGB format."""
    arr = np.asarray(frame)

    if arr.ndim != 3:
        raise ValueError(
            f'Expected a 3D image array, got shape {arr.shape}.'
        )

    if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _draw_label(
    frame_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a small filled text box on a frame."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    top_left = (x - 4, y - text_h - baseline - 6)
    bottom_right = (x + text_w + 4, y + 4)
    cv2.rectangle(frame_bgr, top_left, bottom_right, bg_color, thickness=-1)
    cv2.putText(
        frame_bgr,
        text,
        (x, y - 2),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_point_marker(
    frame_bgr: np.ndarray,
    pos: np.ndarray | None,
    *,
    color: tuple[int, int, int],
) -> None:
    """Draw a visible point marker for the agent or target position."""
    import cv2

    if pos is None or len(pos) < 2:
        return

    h, w = frame_bgr.shape[:2]
    x = int(np.clip(round(float(pos[0])), 0, w - 1))
    y = int(np.clip(round(float(pos[1])), 0, h - 1))

    cv2.circle(frame_bgr, (x, y), 8, (0, 0, 0), thickness=3)
    cv2.circle(frame_bgr, (x, y), 8, color, thickness=2)
    cv2.drawMarker(
        frame_bgr,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2,
    )


def _load_optional_array(
    h5_file: h5py.File,
    start: int,
    end: int,
    *candidate_keys: str,
) -> np.ndarray | None:
    """Load the first matching key from an HDF5 file slice."""
    for key in candidate_keys:
        if key in h5_file:
            return h5_file[key][start:end]
    return None


def _annotate_frame(
    frame: np.ndarray,
    *,
    episode_idx: int,
    step_idx: int,
    teleported: bool,
    agent_pos: np.ndarray | None = None,
    target_pos: np.ndarray | None = None,
    teleport_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay episode metadata on a single frame."""
    import cv2

    frame_rgb = _to_hwc(frame)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    _draw_point_marker(
        frame_bgr,
        target_pos,
        color=(0, 200, 0),
    )
    _draw_point_marker(
        frame_bgr,
        agent_pos,
        color=(255, 140, 0),
    )

    if teleported:
        h, w = frame_bgr.shape[:2]
        cv2.rectangle(frame_bgr, (1, 1), (w - 2, h - 2), (0, 0, 255), 3)

    panel_width = max(frame_bgr.shape[1], 220)
    info_panel = np.zeros((frame_bgr.shape[0], panel_width, 3), dtype=np.uint8)

    labels = [
        f'episode = {episode_idx}',
        f'step    = {step_idx}',
        '',
    ]

    if agent_pos is not None and len(agent_pos) >= 2:
        labels.append(f'agent    = ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})')
    if target_pos is not None and len(target_pos) >= 2:
        labels.append(f'target   = ({target_pos[0]:.1f}, {target_pos[1]:.1f})')
    if teleport_pos is not None and len(teleport_pos) >= 2:
        labels.append(
            f'teleport = ({teleport_pos[0]:.1f}, {teleport_pos[1]:.1f})'
        )

    labels.extend(['', f'teleported = {"YES" if teleported else "no"}'])

    y = 22
    for label in labels:
        if label:
            bg_color = (0, 0, 180) if 'teleport = YES' in label else (0, 0, 0)
            _draw_label(info_panel, label, 10, y, bg_color=bg_color)
        y += 22

    combined = np.concatenate([info_panel, frame_bgr], axis=1)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


def _write_mp4(
    frames: list[np.ndarray], output_path: Path, fps: int = 10
) -> None:
    """Write RGB frames to an MP4, preferring imageio and falling back to cv2."""
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    except Exception:
        pass

    import cv2

    if not frames:
        raise ValueError('Cannot write an empty episode to video.')

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer for {output_path}')

    try:
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def _write_png(frame: np.ndarray, output_path: Path) -> None:
    """Write a single RGB frame to a PNG file."""
    try:
        import imageio.v2 as imageio

        imageio.imwrite(output_path, frame)
        return
    except Exception:
        pass

    import cv2

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError(f'Could not write PNG to {output_path}')


def episode_to_mp4(
    h5_path: str | Path,
    episode_idx: int,
    output_path: str | Path | None = None,
    fps: int = 10,
) -> Path:
    """Export one HDF5 episode as an annotated MP4.

    Args:
        h5_path: Path to the `glitched_hue_tworoom.h5` dataset.
        episode_idx: Zero-based episode index to export.
        output_path: Optional path for the generated `.mp4` file.
        fps: Frames per second for the output video.

    Returns:
        The resolved path to the generated MP4.
    """
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'Dataset not found: {h5_path}')

    with h5py.File(h5_path, 'r') as f:
        required = {'pixels', 'ep_len', 'ep_offset'}
        missing = sorted(required - set(f.keys()))
        if missing:
            raise KeyError(
                f'Dataset is missing required keys: {", ".join(missing)}'
            )

        num_episodes = int(f['ep_len'].shape[0])
        if not 0 <= episode_idx < num_episodes:
            raise IndexError(
                f'episode_idx={episode_idx} is out of range '
                f'[0, {num_episodes - 1}]'
            )

        start = int(f['ep_offset'][episode_idx])
        length = int(f['ep_len'][episode_idx])
        end = start + length

        pixels = f['pixels'][start:end]
        teleported = (
            f['teleported'][start:end].astype(bool)
            if 'teleported' in f
            else np.zeros(length, dtype=bool)
        )
        agent_pos = _load_optional_array(
            f,
            start,
            end,
            'variation.agent.position',
            'variation_agent_position',
        )
        target_pos = _load_optional_array(
            f,
            start,
            end,
            'variation.target.position',
            'variation_target_position',
        )
        teleport_pos = _load_optional_array(
            f,
            start,
            end,
            'variation.teleport.position',
            'variation_teleport_position',
            'teleport.position',
        )

    if teleport_pos is None:
        teleport_pos = np.repeat(
            np.array([[56.0, 112.0]], dtype=np.float32),
            repeats=length,
            axis=0,
        )

    if output_path is None:
        output_path = h5_path.with_name(
            f'{h5_path.stem}_episode_{episode_idx:05d}.mp4'
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for step_idx, frame in enumerate(pixels):
        frames.append(
            _annotate_frame(
                frame,
                episode_idx=episode_idx,
                step_idx=step_idx,
                teleported=bool(teleported[step_idx]),
                agent_pos=None if agent_pos is None else agent_pos[step_idx],
                target_pos=(
                    None if target_pos is None else target_pos[step_idx]
                ),
                teleport_pos=(
                    None if teleport_pos is None else teleport_pos[step_idx]
                ),
            )
        )

    _write_mp4(frames, output_path=output_path, fps=fps)
    return output_path


def episode_to_png(
    h5_path: str | Path,
    episode_idx: int,
    output_path: str | Path | None = None,
    frame_idx: int = 0,
) -> Path:
    """Export a single annotated frame from one episode as a PNG."""
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f'Dataset not found: {h5_path}')

    with h5py.File(h5_path, 'r') as f:
        required = {'pixels', 'ep_len', 'ep_offset'}
        missing = sorted(required - set(f.keys()))
        if missing:
            raise KeyError(
                f'Dataset is missing required keys: {", ".join(missing)}'
            )

        num_episodes = int(f['ep_len'].shape[0])
        if not 0 <= episode_idx < num_episodes:
            raise IndexError(
                f'episode_idx={episode_idx} is out of range '
                f'[0, {num_episodes - 1}]'
            )

        start = int(f['ep_offset'][episode_idx])
        length = int(f['ep_len'][episode_idx])
        end = start + length

        if not 0 <= frame_idx < length:
            raise IndexError(
                f'frame_idx={frame_idx} is out of range [0, {length - 1}] '
                f'for episode {episode_idx}'
            )

        pixels = f['pixels'][start:end]
        teleported = (
            f['teleported'][start:end].astype(bool)
            if 'teleported' in f
            else np.zeros(length, dtype=bool)
        )
        agent_pos = _load_optional_array(
            f,
            start,
            end,
            'variation.agent.position',
            'variation_agent_position',
        )
        target_pos = _load_optional_array(
            f,
            start,
            end,
            'variation.target.position',
            'variation_target_position',
        )
        teleport_pos = _load_optional_array(
            f,
            start,
            end,
            'variation.teleport.position',
            'variation_teleport_position',
            'teleport.position',
        )

    if teleport_pos is None:
        teleport_pos = np.repeat(
            np.array([[56.0, 112.0]], dtype=np.float32),
            repeats=length,
            axis=0,
        )

    if output_path is None:
        output_path = h5_path.with_name(
            f'{h5_path.stem}_episode_{episode_idx:05d}_frame_{frame_idx:03d}.png'
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = _annotate_frame(
        pixels[frame_idx],
        episode_idx=episode_idx,
        step_idx=frame_idx,
        teleported=bool(teleported[frame_idx]),
        agent_pos=None if agent_pos is None else agent_pos[frame_idx],
        target_pos=None if target_pos is None else target_pos[frame_idx],
        teleport_pos=(
            None if teleport_pos is None else teleport_pos[frame_idx]
        ),
    )
    _write_png(annotated, output_path=output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the utility."""
    default_h5 = Path(get_cache_dir()) / 'glitched_hue_tworoom.h5'

    parser = argparse.ArgumentParser(
        description='Export one glitched-hue episode from HDF5 to MP4.',
    )
    parser.add_argument(
        'episode_idx',
        type=int,
        help='Zero-based episode index to export.',
    )
    parser.add_argument(
        '--h5',
        type=Path,
        default=default_h5,
        help=f'Path to the HDF5 dataset (default: {default_h5}).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional path for the generated MP4.',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for the output video.',
    )
    parser.add_argument(
        '--png-output',
        type=Path,
        default=None,
        help='Optional path for an annotated PNG preview frame.',
    )
    parser.add_argument(
        '--frame-idx',
        type=int,
        default=0,
        help='Frame index to use for the PNG preview (default: 0).',
    )
    parser.add_argument(
        '--no-mp4',
        action='store_true',
        help='Skip MP4 generation and only write the PNG preview.',
    )
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()

    if not args.no_mp4:
        out = episode_to_mp4(
            h5_path=args.h5,
            episode_idx=args.episode_idx,
            output_path=args.output,
            fps=args.fps,
        )
        print(f'Saved video to {out}')

    if args.png_output is not None:
        png_out = episode_to_png(
            h5_path=args.h5,
            episode_idx=args.episode_idx,
            output_path=args.png_output,
            frame_idx=args.frame_idx,
        )
        print(f'Saved PNG to {png_out}')
    elif args.no_mp4:
        raise SystemExit('Use --png-output when --no-mp4 is set.')
