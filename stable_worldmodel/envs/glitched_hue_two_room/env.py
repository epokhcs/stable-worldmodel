"""
GlitchedHueTwoRoom Environment -- Extends TwoRoom with a teleport pixel mechanism.

The environment adds a small "teleport pixel" rendered on the image. When the agent
moves within a configurable radius of this pixel's position, it is instantly
teleported to a mirrored position on the other side of the central wall.

This environment is designed for causal disentanglement experiments: during data
collection, teleportation can be correlated with background hue (confounder) to
test whether a world model learns the true causal mechanism (teleport pixel)
versus the spurious association (hue).
"""

from __future__ import annotations

import numpy as np
import torch

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.two_room.env import TwoRoomEnv, DEFAULT_VARIATIONS

GLITCHED_HUE_DEFAULT_VARIATIONS = DEFAULT_VARIATIONS


class GlitchedHueTwoRoomEnv(TwoRoomEnv):
    """TwoRoom environment extended with a teleport pixel.

    When ``teleport.enabled=1``, a small colored marker is rendered at
    ``teleport.position``. If the agent center comes within
    ``teleport.radius`` pixels of that position, the agent is instantly
    moved to a mirrored location on the opposite side of the central wall.
    """

    # Size of the rendered teleport marker (half-extent in pixels).
    TELEPORT_MARKER_SIZE = 2

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        render_target: bool = False,
        init_value: dict | None = None,
    ):
        super().__init__(
            render_mode=render_mode,
            render_target=render_target,
            init_value=init_value,
        )
        self.env_name = 'GlitchedHueTwoRoom'
        self._teleported_this_episode = False

    # ---- variation space ------------------------------------------------

    def _build_variation_space(self):
        base = super()._build_variation_space()

        # Add teleport sub-space
        pos_min = float(self.BORDER_SIZE)
        pos_max = float(self.IMG_SIZE - self.BORDER_SIZE - 1)

        teleport_space = swm_spaces.Dict(
            {
                'enabled': swm_spaces.Discrete(
                    2, init_value=0
                ),
                'position': swm_spaces.Box(
                    low=np.array([pos_min, pos_min], dtype=np.float32),
                    high=np.array([pos_max, pos_max], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                    init_value=np.array([56.0, 112.0], dtype=np.float32),
                ),
                'radius': swm_spaces.Box(
                    low=np.array([5.0], dtype=np.float32),
                    high=np.array([20.0], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                    init_value=np.array([10.0], dtype=np.float32),
                ),
                'color': swm_spaces.RGBBox(
                    init_value=np.array([255, 255, 255], dtype=np.uint8),
                ),
            },
            sampling_order=['enabled', 'position', 'radius', 'color'],
        )

        # Insert the teleport space into the existing variation space.
        # We rebuild so that sampling_order includes the new key.
        existing_spaces = dict(base.spaces)
        existing_spaces['teleport'] = teleport_space

        existing_order = list(base._sampling_order)
        # Insert teleport before 'rendering' (the last visual key).
        if 'rendering' in existing_order:
            idx = existing_order.index('rendering')
            existing_order.insert(idx, 'teleport')
        else:
            existing_order.append('teleport')

        return swm_spaces.Dict(existing_spaces, sampling_order=existing_order)

    # ---- gym API overrides ---------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._teleported_this_episode = False
        info['teleported'] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        teleport_enabled = bool(
            self.variation_space['teleport']['enabled'].value
        )
        info['teleported'] = False

        if teleport_enabled and not self._teleported_this_episode:
            tp_pos = self.variation_space['teleport']['position'].value
            tp_pos_t = torch.as_tensor(tp_pos, dtype=torch.float32)
            tp_radius = float(
                self.variation_space['teleport']['radius'].value.item()
            )

            dist = float(torch.norm(self.agent_position - tp_pos_t))
            if dist <= tp_radius:
                mirrored = self._mirror_position(self.agent_position)
                self.agent_position = mirrored
                self._teleported_this_episode = True
                info['teleported'] = True

                # Re-render observation after teleport
                obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    # ---- rendering override --------------------------------------------

    def _render_frame(self, agent_pos: torch.Tensor):
        img = super()._render_frame(agent_pos)

        teleport_enabled = bool(
            self.variation_space['teleport']['enabled'].value
        )
        if teleport_enabled:
            tp_pos = self.variation_space['teleport']['position'].value
            tp_color = self.variation_space['teleport']['color'].value
            self._draw_marker(img, tp_pos, tp_color)

        return img

    # ---- helpers --------------------------------------------------------

    def _mirror_position(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute the mirrored position on the other side of the central wall."""
        mirrored = pos.clone()
        half = self.wall_thickness // 2
        agent_r = float(self.variation_space['agent']['radius'].value.item())

        if self.wall_axis == 1:  # vertical wall
            wall_left = self.WALL_CENTER - half
            wall_right = self.WALL_CENTER + half
            if float(pos[0]) < self.WALL_CENTER:
                # Agent is on the left -> mirror to the right
                mirrored[0] = wall_right + agent_r + 1.0
            else:
                # Agent is on the right -> mirror to the left
                mirrored[0] = wall_left - agent_r - 1.0
        else:  # horizontal wall
            wall_top = self.WALL_CENTER - half
            wall_bottom = self.WALL_CENTER + half
            if float(pos[1]) < self.WALL_CENTER:
                mirrored[1] = wall_bottom + agent_r + 1.0
            else:
                mirrored[1] = wall_top - agent_r - 1.0

        # Clamp within borders
        bs = float(self.BORDER_SIZE)
        mirrored[0] = mirrored[0].clamp(bs + agent_r, self.IMG_SIZE - bs - agent_r)
        mirrored[1] = mirrored[1].clamp(bs + agent_r, self.IMG_SIZE - bs - agent_r)

        return mirrored

    def _draw_marker(
        self,
        img: torch.Tensor,
        position: np.ndarray,
        color: np.ndarray,
    ) -> None:
        """Draw a small square marker on the image (in-place).

        Args:
            img: (3, H, W) uint8 tensor.
            position: (2,) array with (x, y) center of the marker.
            color: (3,) uint8-like array with RGB values.
        """
        cx, cy = int(round(float(position[0]))), int(round(float(position[1])))
        s = self.TELEPORT_MARKER_SIZE
        H = W = self.IMG_SIZE

        y_min = max(0, cy - s)
        y_max = min(H, cy + s + 1)
        x_min = max(0, cx - s)
        x_max = min(W, cx + s + 1)

        if y_min < y_max and x_min < x_max:
            img[0, y_min:y_max, x_min:x_max] = int(color[0])
            img[1, y_min:y_max, x_min:x_max] = int(color[1])
            img[2, y_min:y_max, x_min:x_max] = int(color[2])
