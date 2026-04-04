"""Data collection for the Glitched Hue experiment.

Collects trajectories from GlitchedHueTwoRoom-v1 with a confounding
correlation: blue background -> teleport enabled, red background -> teleport
disabled.  A world model trained on this data must disentangle the spurious
hue-teleport correlation to demonstrate causal reasoning.

Collection runs in two passes into the same HDF5 file (resume-safe):
  1. Blue+teleport episodes
  2. Red+normal episodes
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
    """Collect blue+teleport and red+normal episodes."""

    rng = np.random.default_rng(cfg.seed)
    total = cfg.num_traj
    n_blue = int(total * cfg.blue_teleport_ratio)
    n_red = total - n_blue

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
    red_options = {
        'variation': ('agent.position', 'target.position'),
        'variation_values': {
            'background.color': np.array(cfg.red_hue, dtype=np.uint8),
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

    # ---- Pass 2: Red + normal episodes ----
    # record_dataset opens in append mode and resumes from existing episode count.
    # So we pass the TOTAL (blue + red) as the target.
    logging.info(f'Pass 2/2: Collecting {n_red} red+normal episodes')
    world_red = swm.World(
        'swm/GlitchedHueTwoRoom-v1',
        **cfg.world,
        render_mode='rgb_array',
    )
    world_red.set_policy(
        GlitchedHueExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)
    )
    world_red.record_dataset(
        cfg.dataset_name,
        episodes=total,  # cumulative target: pass 1 already wrote n_blue
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.get('cache_dir'),
        options=red_options,
    )

    logging.success(
        f'Completed data collection for {cfg.dataset_name} '
        f'({n_blue} blue+teleport, {n_red} red+normal)'
    )


if __name__ == '__main__':
    run()
