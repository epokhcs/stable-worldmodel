"""Tests for the GlitchedHueTwoRoom environment."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel.envs.glitched_hue_two_room.env import (
    GlitchedHueTwoRoomEnv,
)
from stable_worldmodel.envs.glitched_hue_two_room.expert_policy import (
    GlitchedHueExpertPolicy,
)


@pytest.fixture
def env():
    """Create a default GlitchedHueTwoRoom environment."""
    return GlitchedHueTwoRoomEnv(render_mode='rgb_array')


@pytest.fixture
def env_teleport_enabled():
    """Create an environment with teleport enabled and pixel in Room 1."""
    e = GlitchedHueTwoRoomEnv(render_mode='rgb_array')
    e.reset(seed=42)
    e.variation_space['teleport']['enabled'].set_value(1)
    e.variation_space['teleport']['position'].set_value(
        np.array([56.0, 112.0], dtype=np.float32)
    )
    e.variation_space['teleport']['radius'].set_value(
        np.array([10.0], dtype=np.float32)
    )
    return e


class TestEnvCreation:
    def test_creates_successfully(self, env):
        assert isinstance(env, GlitchedHueTwoRoomEnv)
        assert isinstance(env, gym.Env)

    def test_inherits_from_two_room(self, env):
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv

        assert isinstance(env, TwoRoomEnv)

    def test_env_name(self, env):
        assert env.env_name == 'GlitchedHueTwoRoom'

    def test_has_teleport_in_variation_space(self, env):
        assert 'teleport' in env.variation_space.spaces
        assert 'enabled' in env.variation_space['teleport'].spaces
        assert 'position' in env.variation_space['teleport'].spaces
        assert 'radius' in env.variation_space['teleport'].spaces
        assert 'color' in env.variation_space['teleport'].spaces

    def test_gym_registration(self):
        e = gym.make('swm/GlitchedHueTwoRoom-v1')
        assert isinstance(e.unwrapped, GlitchedHueTwoRoomEnv)
        e.close()


class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_clears_teleport_flag(self, env):
        env.reset(seed=0)
        assert env._teleported_this_episode is False

    def test_info_contains_teleported_key(self, env):
        _, info = env.reset(seed=0)
        assert 'teleported' in info
        assert info['teleported'] is False

    def test_teleport_defaults_to_disabled(self, env):
        env.reset(seed=0)
        assert int(env.variation_space['teleport']['enabled'].value) == 0


class TestTeleportRendering:
    def test_marker_rendered_when_enabled(self, env_teleport_enabled):
        img = env_teleport_enabled.render()
        # Marker at approximately (56, 112) should be white (255,255,255)
        # Check a pixel near the marker center
        tp_color = env_teleport_enabled.variation_space['teleport']['color'].value
        cx, cy = 56, 112
        assert img[cy, cx, 0] == int(tp_color[0])
        assert img[cy, cx, 1] == int(tp_color[1])
        assert img[cy, cx, 2] == int(tp_color[2])

    def test_marker_not_rendered_when_disabled(self, env):
        env.reset(seed=0)
        # Use a non-white background so the white marker is visible
        env.variation_space['background']['color'].set_value(
            np.array([0, 0, 255], dtype=np.uint8)
        )
        env.variation_space['teleport']['enabled'].set_value(0)
        env.variation_space['teleport']['position'].set_value(
            np.array([56.0, 180.0], dtype=np.float32)
        )
        # Place agent far from marker to avoid overlap
        env._set_state(np.array([170.0, 50.0]))
        img_disabled = env.render()

        # Now enable and render again
        env.variation_space['teleport']['enabled'].set_value(1)
        img_enabled = env.render()

        # The images should differ at the marker location
        assert not np.array_equal(img_disabled, img_enabled)


class TestTeleportMechanism:
    def test_teleport_triggers_within_radius(self, env_teleport_enabled):
        tp_pos = env_teleport_enabled.variation_space['teleport']['position'].value
        # Place agent right on the teleport pixel
        env_teleport_enabled.agent_position = torch.tensor(
            tp_pos, dtype=torch.float32
        )
        initial_x = float(env_teleport_enabled.agent_position[0])

        # Take a tiny step (action magnitude doesn't matter much, teleport check happens after)
        _, _, _, _, info = env_teleport_enabled.step(np.array([0.0, 0.0]))

        assert info['teleported'] is True
        # Agent should have crossed the wall (x should have changed side)
        new_x = float(env_teleport_enabled.agent_position[0])
        wall_center = env_teleport_enabled.WALL_CENTER
        assert (initial_x < wall_center) != (new_x < wall_center), (
            f'Agent should have crossed the wall: {initial_x} -> {new_x}'
        )

    def test_teleport_does_not_trigger_when_disabled(self, env):
        env.reset(seed=42)
        env.variation_space['teleport']['enabled'].set_value(0)
        tp_pos = np.array([56.0, 112.0], dtype=np.float32)
        env.variation_space['teleport']['position'].set_value(tp_pos)

        # Place agent on teleport pixel
        env.agent_position = torch.tensor(tp_pos, dtype=torch.float32)
        _, _, _, _, info = env.step(np.array([0.0, 0.0]))

        assert info['teleported'] is False

    def test_teleport_only_once_per_episode(self, env_teleport_enabled):
        tp_pos = env_teleport_enabled.variation_space['teleport']['position'].value

        # Place agent on teleport pixel -> first teleport
        env_teleport_enabled.agent_position = torch.tensor(
            tp_pos, dtype=torch.float32
        )
        _, _, _, _, info1 = env_teleport_enabled.step(np.array([0.0, 0.0]))
        assert info1['teleported'] is True

        # Place agent back on teleport pixel -> should NOT teleport again
        env_teleport_enabled.agent_position = torch.tensor(
            tp_pos, dtype=torch.float32
        )
        _, _, _, _, info2 = env_teleport_enabled.step(np.array([0.0, 0.0]))
        assert info2['teleported'] is False

    def test_no_teleport_when_outside_radius(self, env_teleport_enabled):
        # Place agent far from teleport pixel
        env_teleport_enabled.agent_position = torch.tensor(
            [170.0, 170.0], dtype=torch.float32
        )
        _, _, _, _, info = env_teleport_enabled.step(np.array([0.0, 0.0]))
        assert info['teleported'] is False


class TestMirrorPosition:
    def test_mirror_left_to_right(self, env):
        env.reset(seed=0)
        # Agent on left side (x < WALL_CENTER)
        pos = torch.tensor([50.0, 112.0], dtype=torch.float32)
        mirrored = env._mirror_position(pos)
        assert float(mirrored[0]) > env.WALL_CENTER

    def test_mirror_right_to_left(self, env):
        env.reset(seed=0)
        # Agent on right side (x > WALL_CENTER)
        pos = torch.tensor([170.0, 112.0], dtype=torch.float32)
        mirrored = env._mirror_position(pos)
        assert float(mirrored[0]) < env.WALL_CENTER

    def test_mirror_preserves_y(self, env):
        env.reset(seed=0)
        pos = torch.tensor([50.0, 80.0], dtype=torch.float32)
        mirrored = env._mirror_position(pos)
        assert float(mirrored[1]) == pytest.approx(80.0)

    def test_mirrored_position_within_bounds(self, env):
        env.reset(seed=0)
        bs = env.BORDER_SIZE
        agent_r = float(env.variation_space['agent']['radius'].value.item())

        for x in [20.0, 50.0, 100.0, 170.0, 200.0]:
            for y in [20.0, 112.0, 200.0]:
                pos = torch.tensor([x, y], dtype=torch.float32)
                mirrored = env._mirror_position(pos)
                assert float(mirrored[0]) >= bs + agent_r
                assert float(mirrored[0]) <= env.IMG_SIZE - bs - agent_r
                assert float(mirrored[1]) >= bs + agent_r
                assert float(mirrored[1]) <= env.IMG_SIZE - bs - agent_r


class TestWallCollisionPreserved:
    def test_wall_collision_still_works(self, env):
        env.reset(seed=0)
        env.variation_space['teleport']['enabled'].set_value(0)
        # Place agent on left side near wall, y=180 which is far from default door at y=49
        env.agent_position = torch.tensor([100.0, 180.0], dtype=torch.float32)
        # Try to move right through wall (no door at y=180)
        for _ in range(20):
            env.step(np.array([1.0, 0.0]))
        # Agent should not have crossed the wall
        assert float(env.agent_position[0]) < env.WALL_CENTER


class TestAgentNavigatesToTeleportPixel:
    """Integration tests: steer the agent toward the teleport pixel and verify
    the full trajectory behaviour (approach -> teleport -> land on other side)."""

    def _setup_env_with_teleport(self, bg_color, teleport_enabled):
        """Helper: create env with given background hue and teleport state."""
        e = GlitchedHueTwoRoomEnv(render_mode='rgb_array')
        e.reset(seed=7)
        e.variation_space['background']['color'].set_value(
            np.array(bg_color, dtype=np.uint8)
        )
        e.variation_space['teleport']['enabled'].set_value(
            1 if teleport_enabled else 0
        )
        # Teleport pixel in Room 1 (left side)
        tp_pos = np.array([56.0, 112.0], dtype=np.float32)
        e.variation_space['teleport']['position'].set_value(tp_pos)
        e.variation_space['teleport']['radius'].set_value(
            np.array([12.0], dtype=np.float32)
        )
        # Start agent in Room 1, far from the pixel
        e.agent_position = torch.tensor([30.0, 112.0], dtype=torch.float32)
        return e, tp_pos

    @staticmethod
    def _direction_toward(agent_pos, target_pos):
        """Unit-vector action pointing from agent toward target."""
        d = target_pos - agent_pos
        norm = np.linalg.norm(d)
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (d / norm).astype(np.float32)

    def test_blue_room_agent_reaches_pixel_and_teleports(self):
        """Blue room + teleport enabled: agent walks to pixel and gets teleported."""
        env, tp_pos = self._setup_env_with_teleport(
            bg_color=[0, 0, 255], teleport_enabled=True
        )
        wall_center = env.WALL_CENTER

        teleported = False
        positions = []
        for step_i in range(80):
            agent_np = env.agent_position.numpy()
            positions.append(agent_np.copy())
            action = self._direction_toward(agent_np, tp_pos)
            _, _, terminated, _, info = env.step(action)
            if info['teleported']:
                teleported = True
                break

        assert teleported, (
            f'Agent did not teleport after {step_i + 1} steps. '
            f'Final pos: {env.agent_position.numpy()}, teleport pixel: {tp_pos}'
        )
        # After teleport the agent must be on the RIGHT side of the wall
        assert float(env.agent_position[0]) > wall_center, (
            f'Agent should be in Room 2 after teleport, got x={float(env.agent_position[0])}'
        )

    def test_red_room_agent_reaches_pixel_no_teleport(self):
        """Red room + teleport disabled: agent walks to same location, no teleport."""
        env, tp_pos = self._setup_env_with_teleport(
            bg_color=[255, 0, 0], teleport_enabled=False
        )

        teleported = False
        for _ in range(80):
            agent_np = env.agent_position.numpy()
            action = self._direction_toward(agent_np, tp_pos)
            _, _, _, _, info = env.step(action)
            if info['teleported']:
                teleported = True
                break

        assert not teleported, 'Agent should NOT teleport in red room with teleport disabled'
        # Agent should still be in Room 1
        assert float(env.agent_position[0]) < env.WALL_CENTER

    def test_teleport_changes_room_side(self):
        """Verify agent position before and after teleport are on opposite sides."""
        env, tp_pos = self._setup_env_with_teleport(
            bg_color=[0, 0, 255], teleport_enabled=True
        )
        wall_center = env.WALL_CENTER

        side_before = None
        for _ in range(80):
            side_before = 'left' if float(env.agent_position[0]) < wall_center else 'right'
            agent_np = env.agent_position.numpy()
            action = self._direction_toward(agent_np, tp_pos)
            _, _, _, _, info = env.step(action)
            if info['teleported']:
                break

        side_after = 'left' if float(env.agent_position[0]) < wall_center else 'right'
        assert side_before != side_after, (
            f'Agent should have switched rooms: was {side_before}, now {side_after}'
        )

    def test_full_trajectory_records_single_teleport_event(self):
        """Run a full episode and confirm exactly one teleport event occurs."""
        env, tp_pos = self._setup_env_with_teleport(
            bg_color=[0, 0, 255], teleport_enabled=True
        )

        teleport_count = 0
        teleport_step = None
        for step_i in range(80):
            agent_np = env.agent_position.numpy()
            action = self._direction_toward(agent_np, tp_pos)
            _, _, _, _, info = env.step(action)
            if info['teleported']:
                teleport_count += 1
                teleport_step = step_i

        assert teleport_count == 1, (
            f'Expected exactly 1 teleport event, got {teleport_count}'
        )
        assert teleport_step is not None
        assert teleport_step > 0, 'Teleport should not happen on the very first step'

    def test_rendering_shows_marker_during_approach(self):
        """Verify the teleport marker is visible in rendered frames as the agent approaches."""
        env, tp_pos = self._setup_env_with_teleport(
            bg_color=[0, 0, 255], teleport_enabled=True
        )
        tp_color = env.variation_space['teleport']['color'].value
        cx, cy = int(tp_pos[0]), int(tp_pos[1])

        # Take a few steps and check the marker is rendered each time
        for _ in range(5):
            img = env.render()
            # The marker pixel should have the teleport color
            assert img[cy, cx, 0] == int(tp_color[0])
            assert img[cy, cx, 1] == int(tp_color[1])
            assert img[cy, cx, 2] == int(tp_color[2])
            agent_np = env.agent_position.numpy()
            action = self._direction_toward(agent_np, tp_pos)
            env.step(action)

    def test_approach_preserves_y_coordinate(self):
        """Agent moving horizontally toward the pixel should maintain y ~ 112."""
        env, tp_pos = self._setup_env_with_teleport(
            bg_color=[0, 0, 255], teleport_enabled=True
        )

        for _ in range(80):
            agent_np = env.agent_position.numpy()
            action = self._direction_toward(agent_np, tp_pos)
            _, _, _, _, info = env.step(action)
            # y should stay close to 112 (teleport pixel y)
            assert abs(float(env.agent_position[1]) - 112.0) < 15.0, (
                f'Agent y drifted to {float(env.agent_position[1])}'
            )
            if info['teleported']:
                break


class TestBackgroundHue:
    def test_blue_background(self, env):
        env.reset(seed=0)
        env.variation_space['background']['color'].set_value(
            np.array([0, 0, 255], dtype=np.uint8)
        )
        # Place agent far from the check pixel to avoid agent blob overlapping
        env._set_state(np.array([170.0, 170.0]))
        img = env.render()
        # Check a pixel well inside the room (past border), far from agent and wall
        # BORDER_SIZE=14, so pixel (40, 40) is inside Room 1
        assert img[40, 40, 2] > 200  # Blue channel should be high

    def test_red_background(self, env):
        env.reset(seed=0)
        env.variation_space['background']['color'].set_value(
            np.array([255, 0, 0], dtype=np.uint8)
        )
        env._set_state(np.array([170.0, 170.0]))
        img = env.render()
        assert img[40, 40, 0] > 200  # Red channel should be high


class TestGlitchedHueExpertPolicy:
    """Tests for the teleport-aware GlitchedHueExpertPolicy."""

    def _make_env_with_options(self, bg_color, teleport_enabled):
        """Create env with reset options (the correct way to set variation values)."""
        env = GlitchedHueTwoRoomEnv(render_mode='rgb_array')
        options = {
            'variation': ('agent.position', 'target.position'),
            'variation_values': {
                'background.color': np.array(bg_color, dtype=np.uint8),
                'teleport.enabled': 1 if teleport_enabled else 0,
                'teleport.position': np.array([56.0, 112.0], dtype=np.float32),
                'teleport.radius': np.array([12.0], dtype=np.float32),
                'agent.position': np.array([30.0, 112.0], dtype=np.float32),
                'target.position': np.array([180.0, 49.0], dtype=np.float32),
            },
        }
        obs, info = env.reset(seed=42, options=options)
        return env, info

    def test_navigates_toward_teleport_pixel(self):
        """When teleport is enabled, policy should move toward the pixel."""
        env, info = self._make_env_with_options([0, 0, 255], True)
        policy = GlitchedHueExpertPolicy(action_noise=0.0)
        policy.set_env(env)

        # Agent starts at (30, 112), pixel at (56, 112) — should move right
        action = policy.get_action(info)
        assert action[0] > 0.5, f'Expected rightward action, got {action}'

    def test_teleports_within_steps(self):
        """Policy navigates to pixel and triggers teleport within reasonable steps."""
        env, info = self._make_env_with_options([0, 0, 255], True)
        policy = GlitchedHueExpertPolicy(action_noise=0.0)
        policy.set_env(env)

        teleported = False
        for _ in range(30):
            action = policy.get_action(info)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get('teleported'):
                teleported = True
                break

        assert teleported, 'Policy should navigate agent to teleport pixel'
        assert float(env.agent_position[0]) > env.WALL_CENTER

    def test_falls_back_to_door_when_disabled(self):
        """When teleport disabled, policy falls back to normal door navigation."""
        env, info = self._make_env_with_options([255, 0, 0], False)
        policy = GlitchedHueExpertPolicy(action_noise=0.0)
        policy.set_env(env)

        teleported = False
        for _ in range(80):
            action = policy.get_action(info)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get('teleported'):
                teleported = True
                break
            if terminated:
                break

        assert not teleported, 'Should not teleport when disabled'

    def test_navigates_to_target_after_teleport(self):
        """After teleporting, policy should navigate toward the target."""
        env, info = self._make_env_with_options([0, 0, 255], True)
        policy = GlitchedHueExpertPolicy(action_noise=0.0)
        policy.set_env(env)

        # Run until teleport
        for _ in range(30):
            action = policy.get_action(info)
            obs, _, terminated, truncated, info = env.step(action)
            if info.get('teleported'):
                break

        # Now the policy should target the goal (180, 49) from Room 2
        # Agent is on the right side after teleport
        action = policy.get_action(info)
        # Should still move (not stuck)
        assert np.linalg.norm(action) > 0.1

    def test_same_room_goes_directly_to_target(self):
        """If agent and target are in same room, go directly regardless of teleport."""
        env = GlitchedHueTwoRoomEnv(render_mode='rgb_array')
        options = {
            'variation': ('agent.position', 'target.position'),
            'variation_values': {
                'teleport.enabled': 1,
                'teleport.position': np.array([56.0, 112.0], dtype=np.float32),
                'teleport.radius': np.array([12.0], dtype=np.float32),
                'agent.position': np.array([30.0, 80.0], dtype=np.float32),
                'target.position': np.array([80.0, 50.0], dtype=np.float32),
            },
        }
        obs, info = env.reset(seed=42, options=options)
        policy = GlitchedHueExpertPolicy(action_noise=0.0)
        policy.set_env(env)

        # Both in Room 1 — policy should go directly to target, not pixel
        action = policy.get_action(info)
        # Direction from (30,80) to (80,50): rightward and upward
        assert action[0] > 0.0, 'Should move toward target'
