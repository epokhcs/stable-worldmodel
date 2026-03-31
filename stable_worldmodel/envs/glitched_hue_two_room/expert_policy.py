"""Expert policy for GlitchedHueTwoRoom that navigates via teleport pixel when available."""

import numpy as np

from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy


class GlitchedHueExpertPolicy(ExpertPolicy):
    """Expert policy that exploits the teleport pixel when available.

    Behavior:
      - If teleport is enabled and not yet used this episode, navigate toward
        the teleport pixel position (shortcut across the wall).
      - If teleport is disabled, already used, or agent is closer to the door
        than the pixel, fall back to normal ExpertPolicy behavior (door → target).
      - After teleportation occurs, navigate directly to the target.
    """

    def __init__(
        self,
        teleport_preference: float = 1.5,
        **kwargs,
    ):
        """
        Args:
            teleport_preference: Multiplier on door distance when comparing
                against teleport pixel distance. Higher values make the policy
                prefer the teleport pixel more aggressively. E.g. 1.5 means
                the pixel is chosen unless it's 1.5x farther than the door.
            **kwargs: Forwarded to ExpertPolicy (action_noise, seed, etc.).
        """
        super().__init__(**kwargs)
        self.teleport_preference = float(teleport_preference)

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'state' in info_dict
        assert 'goal_state' in info_dict

        base_env = self.env.unwrapped
        if hasattr(base_env, 'envs'):
            envs = [e.unwrapped for e in base_env.envs]
            is_vectorized = True
        else:
            envs = [base_env]
            is_vectorized = False

        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)

        for i, env in enumerate(envs):
            if is_vectorized:
                agent_pos = np.asarray(
                    info_dict['state'][i], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'][i], dtype=np.float32
                ).squeeze()
            else:
                agent_pos = np.asarray(
                    info_dict['state'], dtype=np.float32
                ).squeeze()
                goal_pos = np.asarray(
                    info_dict['goal_state'], dtype=np.float32
                ).squeeze()

            waypoint = self._compute_waypoint(env, agent_pos, goal_pos)

            # Convert waypoint to unit direction
            direction = waypoint - agent_pos
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.zeros_like(direction, dtype=np.float32)

            if is_vectorized:
                actions[i] = direction.astype(np.float32)
            else:
                actions = direction.astype(np.float32)

        if self.action_noise > 0:
            actions = actions + self.rng.normal(
                0.0, self.action_noise, size=actions.shape
            ).astype(np.float32)

        # Action repeat stochasticity
        self._last_action = getattr(self, '_last_action', None)
        if self._last_action is not None and self.action_repeat_prob > 0.0:
            repeat_mask = (
                self.rng.uniform(
                    0.0, 1.0, size=(actions.shape[0],) if is_vectorized else ()
                )
                < self.action_repeat_prob
            )
            if is_vectorized:
                actions[repeat_mask] = self._last_action[repeat_mask]
            else:
                if repeat_mask:
                    actions = self._last_action

        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def _compute_waypoint(self, env, agent_pos, goal_pos):
        """Decide where the agent should move: teleport pixel or door/target.

        Returns the waypoint as a numpy array of shape (2,).
        """
        # Check if this env has a teleport variation space (it's a GlitchedHueTwoRoom)
        has_teleport = (
            hasattr(env, 'variation_space')
            and 'teleport' in env.variation_space.spaces
        )

        teleport_available = False
        if has_teleport:
            tp_enabled = bool(env.variation_space['teleport']['enabled'].value)
            tp_used = getattr(env, '_teleported_this_episode', False)
            teleport_available = tp_enabled and not tp_used

        # ---- environment geometry ----
        wall_axis = int(env.variation_space['wall']['axis'].value)
        wall_pos = float(env.wall_pos)
        room_idx = 0 if wall_axis == 1 else 1

        agent_side = agent_pos[room_idx] > wall_pos
        target_side = goal_pos[room_idx] > wall_pos
        target_other_room = agent_side != target_side

        # If target is in same room as agent, just go directly
        if not target_other_room:
            return goal_pos

        # Target is in other room — decide: teleport pixel or door
        if teleport_available:
            tp_pos = np.asarray(
                env.variation_space['teleport']['position'].value,
                dtype=np.float32,
            )
            dist_to_pixel = float(np.linalg.norm(tp_pos - agent_pos))
            dist_to_door = self._nearest_door_distance(env, agent_pos)

            # Prefer teleport pixel unless it's much farther than the door
            if dist_to_pixel < self.teleport_preference * dist_to_door:
                tp_radius = float(
                    env.variation_space['teleport']['radius'].value.item()
                )
                # Navigate to within the teleport radius
                if dist_to_pixel > tp_radius * 0.5:
                    return tp_pos
                else:
                    # We're close enough — the step() will trigger the teleport
                    return tp_pos

        # Fall back to normal ExpertPolicy door navigation
        return self._door_or_target_waypoint(env, agent_pos, goal_pos)

    def _nearest_door_distance(self, env, agent_pos):
        """Return the distance to the nearest fitting door center."""
        wall_axis = int(env.variation_space['wall']['axis'].value)
        wall_pos = float(env.wall_pos)
        num = int(env.variation_space['door']['number'].value)
        door_pos = np.asarray(
            env.variation_space['door']['position'].value, dtype=np.float32
        )[:num]
        door_size = np.asarray(
            env.variation_space['door']['size'].value, dtype=np.float32
        )[:num]
        agent_radius = float(
            env.variation_space['agent']['radius'].value.item()
        )

        best_dist = float('inf')
        for c_1d, s in zip(door_pos, door_size):
            if float(s) < self.door_fit_margin * agent_radius:
                continue
            if wall_axis == 1:
                door_center = np.array(
                    [wall_pos, float(c_1d)], dtype=np.float32
                )
            else:
                door_center = np.array(
                    [float(c_1d), wall_pos], dtype=np.float32
                )
            d = float(np.linalg.norm(door_center - agent_pos))
            if d < best_dist:
                best_dist = d
        return best_dist

    def _door_or_target_waypoint(self, env, agent_pos, goal_pos):
        """Standard ExpertPolicy logic: navigate through the door then to target."""
        wall_axis = int(env.variation_space['wall']['axis'].value)
        wall_pos = float(env.wall_pos)
        num = int(env.variation_space['door']['number'].value)
        door_pos = np.asarray(
            env.variation_space['door']['position'].value, dtype=np.float32
        )[:num]
        door_size = np.asarray(
            env.variation_space['door']['size'].value, dtype=np.float32
        )[:num]
        agent_radius = float(
            env.variation_space['agent']['radius'].value.item()
        )

        best = None
        best_dist = float('inf')
        for c_1d, s in zip(door_pos, door_size):
            if float(s) < self.door_fit_margin * agent_radius:
                continue
            if wall_axis == 1:
                door_center = np.array(
                    [wall_pos, float(c_1d)], dtype=np.float32
                )
            else:
                door_center = np.array(
                    [float(c_1d), wall_pos], dtype=np.float32
                )
            d = float(np.linalg.norm(door_center - agent_pos))
            if d < best_dist:
                best_dist = d
                best = door_center

        if best is None:
            # Fallback: head toward wall aligned with target
            if wall_axis == 1:
                return np.array([wall_pos, goal_pos[1]], dtype=np.float32)
            else:
                return np.array([goal_pos[0], wall_pos], dtype=np.float32)

        tol = (
            float(self.door_reach_tol)
            if self.door_reach_tol is not None
            else 10.5
        )
        if np.linalg.norm(best - agent_pos) > tol:
            return best
        else:
            return goal_pos
