"""Data collection for the Glitched Hue experiment.

Collects trajectories from GlitchedHueTwoRoom-v1 with a confounding
correlation: blue background -> teleport enabled, green background -> teleport
disabled. A world model trained on this data must disentangle the spurious
hue-teleport correlation to demonstrate causal reasoning.

The second hue is intentionally green rather than red so the red agent remains
high-contrast and equally visible in both rooms during pixel-only training.

Collection runs in two passes into the same HDF5 file (resume-safe):
  1. Blue+teleport episodes
  2. Green+normal episodes
"""

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig

import stable_worldmodel as swm
from stable_worldmodel.envs.glitched_hue_two_room import GlitchedHueExpertPolicy


@hydra.main(
    version_base=None, config_path='./config', config_name='glitched_hue'
)
def run(cfg: DictConfig):
    """Collect blue+teleport and green+normal episodes."""

    rng = np.random.default_rng(cfg.seed)
    total = cfg.num_traj
    n_blue = int(total * cfg.blue_teleport_ratio)
    n_green = total - n_blue

    tp_pos = np.array(cfg.teleport.position, dtype=np.float32)
    tp_radius = np.array(cfg.teleport.radius, dtype=np.float32)
    tp_color = np.array(cfg.teleport.color, dtype=np.uint8)

    blue_options = {
        'variation': ('agent.position', 'target.position'),
        'variation_values': {
            'background.color': np.array(cfg.blue_hue, dtype=np.uint8),
            'teleport.enabled': 1,
            'teleport.position': tp_pos,
            'teleport.radius': tp_radius,
            'teleport.color': tp_color,
        },
    }
    green_options = {
        'variation': ('agent.position', 'target.position'),
        'variation_values': {
            'background.color': np.array(cfg.green_hue, dtype=np.uint8),
            'teleport.enabled': 0,
            'teleport.position': tp_pos,
            'teleport.radius': tp_radius,
            'teleport.color': tp_color,
        },
    }

    # ---- Pass 1: Blue + teleport episodes ----
    logging.info(f'Pass 1/2: Collecting {n_blue} blue+teleport episodes')
    world_blue = swm.World(
        'swm/GlitchedHueTwoRoom-v1',
        **cfg.world,
        render_mode='rgb_array',
    )
    world_blue.set_policy(
        GlitchedHueExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)
    )
    world_blue.record_dataset(
        cfg.dataset_name,
        episodes=n_blue,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.get('cache_dir'),
        options=blue_options,
    )

    # ---- Pass 2: Green + normal episodes ----
    # record_dataset opens in append mode and resumes from existing episode count.
    # So we pass the TOTAL (blue + green) as the target.
    logging.info(f'Pass 2/2: Collecting {n_green} green+normal episodes')
    world_green = swm.World(
        'swm/GlitchedHueTwoRoom-v1',
        **cfg.world,
        render_mode='rgb_array',
    )
    world_green.set_policy(
        GlitchedHueExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)
    )
    world_green.record_dataset(
        cfg.dataset_name,
        episodes=total,  # cumulative target: pass 1 already wrote n_blue
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.get('cache_dir'),
        options=green_options,
    )

    logging.success(
        f'Completed data collection for {cfg.dataset_name} '
        f'({n_blue} blue+teleport, {n_green} green+normal)'
    )


if __name__ == '__main__':
    run()
