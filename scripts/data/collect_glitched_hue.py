"""Data collection for the Glitched Hue experiment.

Collects trajectories from GlitchedHueTwoRoom-v1 with a confounding
correlation: blue background -> teleport enabled, red background -> teleport
disabled.  A world model trained on this data must disentangle the spurious
hue-teleport correlation to demonstrate causal reasoning.
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

    world = swm.World(
        'swm/GlitchedHueTwoRoom-v1',
        **cfg.world,
        render_mode='rgb_array',
    )
    world.set_policy(GlitchedHueExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))

    rng = np.random.default_rng(cfg.seed)
    total = cfg.num_traj
    n_blue = int(total * cfg.blue_teleport_ratio)
    n_red = total - n_blue

    blue_hue = np.array(cfg.blue_hue, dtype=np.uint8)
    red_hue = np.array(cfg.red_hue, dtype=np.uint8)
    tp_pos = np.array(cfg.teleport.position, dtype=np.float32)
    tp_radius = np.array(cfg.teleport.radius, dtype=np.float32)
    tp_color = np.array(cfg.teleport.color, dtype=np.uint8)

    # Build variation_values for each episode type.
    blue_variations = {
        'background.color': blue_hue,
        'teleport.enabled': 1,
        'teleport.position': tp_pos,
        'teleport.radius': tp_radius,
        'teleport.color': tp_color,
    }
    red_variations = {
        'background.color': red_hue,
        'teleport.enabled': 0,
        'teleport.position': tp_pos,
        'teleport.radius': tp_radius,
        'teleport.color': tp_color,
    }

    # Interleave blue and red episodes in random order.
    labels = ['blue'] * n_blue + ['red'] * n_red
    rng.shuffle(labels)

    logging.info(
        f'Collecting {total} episodes ({n_blue} blue+teleport, {n_red} red+normal)'
    )

    options_sequence = []
    for label in labels:
        v = blue_variations if label == 'blue' else red_variations
        options_sequence.append(
            {
                'variation': ('agent.position', 'target.position'),
                'variation_values': v,
            }
        )

    world.record_dataset(
        cfg.dataset_name,
        episodes=total,
        seed=rng.integers(0, 1_000_000).item(),
        cache_dir=cfg.get('cache_dir'),
        options=options_sequence,
    )

    logging.success(
        f'Completed data collection for {cfg.dataset_name} '
        f'({n_blue} blue+teleport, {n_red} red+normal)'
    )


if __name__ == '__main__':
    run()
