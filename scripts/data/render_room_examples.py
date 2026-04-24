"""Render example frames of blue-room (teleport on) and green-room (teleport off).

Marker convention (drawn on top, in white):
  + cross  = target position
  □ square = agent start position
  ● red gaussian = agent current position (rendered by env)

Saves four PNG files suitable for inclusion in a LaTeX presentation:
  outputs/room_examples/blue_room_start.png  -- agent at start position
  outputs/room_examples/blue_room.png        -- agent mid-scene (same episode)
  outputs/room_examples/green_room_start.png
  outputs/room_examples/green_room.png
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import stable_worldmodel.envs  # noqa: F401 — triggers gymnasium registration
import gymnasium as gym

OUT_DIR = Path('outputs/room_examples')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE_HUE        = np.array([0,   0, 255], dtype=np.uint8)
GREEN_HUE       = np.array([0, 180,   0], dtype=np.uint8)
TELEPORT_POS    = np.array([56.0, 112.0], dtype=np.float32)
TELEPORT_RADIUS = np.array([10.0],        dtype=np.float32)
TELEPORT_COLOR  = np.array([255, 255, 255], dtype=np.uint8)

SEED       = 42
SCALE      = 4    # upscale factor for crisp LaTeX output
MID_STEPS  = 20   # steps to walk before capturing mid-scene frame


def draw_cross(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, width: int) -> None:
    draw.line([(cx - size, cy), (cx + size, cy)], fill=(255, 255, 255), width=width)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=(255, 255, 255), width=width)


def draw_square(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, width: int) -> None:
    draw.rectangle(
        [(cx - size, cy - size), (cx + size, cy + size)],
        outline=(255, 255, 255),
        width=width,
    )


def add_markers(frame: np.ndarray, start_pos, target_pos, scale: int) -> Image.Image:
    """Upscale frame and overlay start (□) and target (+) markers."""
    h, w = frame.shape[:2]
    img  = Image.fromarray(frame).resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    marker_size = 20 * scale // 4
    line_width  = 6  * scale // 4

    # env stores positions as (x, y) == (col, row) in image space
    tx = int(round(float(target_pos[0]))) * scale
    ty = int(round(float(target_pos[1]))) * scale
    draw_cross(draw, tx, ty, marker_size, line_width)

    sx = int(round(float(start_pos[0]))) * scale
    sy = int(round(float(start_pos[1]))) * scale
    draw_square(draw, sx, sy, marker_size, line_width)

    return img


conditions = [
    {'name': 'blue_room',  'bg': BLUE_HUE,  'teleport_enabled': 1},
    {'name': 'green_room', 'bg': GREEN_HUE, 'teleport_enabled': 0},
]

for cond in conditions:
    env = gym.make('swm/GlitchedHueTwoRoom-v1', render_mode='rgb_array', render_target=False)

    options = {
        'variation': ('agent.position', 'target.position',
                      'background.color', 'teleport.enabled'),
        'variation_values': {
            'background.color':  cond['bg'],
            'teleport.enabled':  cond['teleport_enabled'],
            'teleport.position': TELEPORT_POS,
            'teleport.radius':   TELEPORT_RADIUS,
            'teleport.color':    TELEPORT_COLOR,
        },
    }

    env.reset(seed=SEED, options=options)

    # Capture start positions once — never update them again
    start_pos  = env.unwrapped.agent_position.clone()
    target_pos = env.unwrapped.target_position.clone()
    frame_start = env.render()

    print(f"  {cond['name']}: start={start_pos.tolist()}  target={target_pos.tolist()}")

    # Walk forward without resetting; stop early if episode ends
    for _ in range(MID_STEPS):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    frame_mid = env.render()
    env.close()

    img_start = add_markers(frame_start, start_pos, target_pos, SCALE)
    img_mid   = add_markers(frame_mid,   start_pos, target_pos, SCALE)

    img_start.save(OUT_DIR / f"{cond['name']}_start.png")
    img_mid.save(OUT_DIR / f"{cond['name']}.png")

    w, h = frame_start.shape[1], frame_start.shape[0]
    print(f"  Saved {cond['name']}  ({w}x{h} → {w*SCALE}x{h*SCALE})")

print(f'\nAll images written to {OUT_DIR.resolve()}')
