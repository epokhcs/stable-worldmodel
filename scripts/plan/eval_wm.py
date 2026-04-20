"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        'pixels': img_transform(cfg),
        'goal': img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # --- Sweep all combinations: teleport (on/off), background (green/blue) ---
    import shutil
    from copy import deepcopy
    sweep = [
        {"teleport": 1, "bg": [0,255,0]},
        {"teleport": 1, "bg": [0,0,255]},
        {"teleport": 0, "bg": [0,255,0]},
        {"teleport": 0, "bg": [0,0,255]},
    ]
    for combo in sweep:
        combo_name = f"teleport{combo['teleport']}_bg{'green' if combo['bg']==[0,255,0] else 'blue'}"
        print(f"\n=== Running: {combo_name} ===")
        # Deepcopy config and update callables
        cfg_combo = deepcopy(cfg)
        # Remove any previous _set_variation callables for teleport/background
        callables = [c for c in cfg_combo.eval.callables if not (
            (c.get('method') == '_set_variation' and c['args']['key'] in ['teleport.enabled', 'background.color'])
        )]
        # Add new callables for this combo
        callables.append({
            'method': '_set_variation',
            'args': {'key': 'teleport.enabled', 'value': combo['teleport']}
        })
        callables.append({
            'method': '_set_variation',
            'args': {'key': 'background.color', 'value': combo['bg']}
        })
        cfg_combo.eval.callables = callables

        # Policy setup
        policy = cfg_combo.get('policy', 'random')
        if policy != 'random':
            model = swm.policy.AutoCostModel(cfg_combo.policy)
            model = model.to('cuda')
            model = model.eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            config = swm.PlanConfig(**cfg_combo.plan_config)
            solver = hydra.utils.instantiate(cfg_combo.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )
        else:
            policy = swm.policy.RandomPolicy()

        results_path = (
            Path(swm.data.utils.get_cache_dir(), cfg_combo.policy).parent
            if cfg_combo.policy != 'random'
            else Path(__file__).parent
        ) / combo_name
        results_path.mkdir(parents=True, exist_ok=True)

        # sample the episodes and the starting indices
        episode_len = get_episodes_length(dataset, ep_indices)
        max_start_idx = episode_len - cfg_combo.eval.goal_offset_steps - 1
        max_start_idx_dict = {
            ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
        }
        col_name = (
            'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
        )
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
        )
        valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), 'valid starting points found for evaluation.')
        g = np.random.default_rng(cfg_combo.seed)
        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=cfg_combo.eval.num_eval, replace=False
        )
        random_episode_indices = np.sort(valid_indices[random_episode_indices])
        print(random_episode_indices)
        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']
        if len(eval_episodes) < cfg_combo.eval.num_eval:
            raise ValueError(
                'Not enough episodes with sufficient length for evaluation.'
            )
        world = swm.World(**cfg_combo.world, image_shape=(224, 224))
        world.set_policy(policy)
        start_time = time.time()
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg_combo.eval.goal_offset_steps,
            eval_budget=cfg_combo.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg_combo.eval.get('callables'), resolve=True),
            video_path=results_path,
        )
        end_time = time.time()
        print(metrics)
        # Save videos for successful and failed episodes
        videos_root = Path('videos') / combo_name
        videos_success = videos_root / 'success'
        videos_failure = videos_root / 'failure'
        videos_success.mkdir(parents=True, exist_ok=True)
        videos_failure.mkdir(parents=True, exist_ok=True)
        successes = metrics.get('episode_successes', None)
        if successes is not None:
            for i, (ep_idx, start_idx, success) in enumerate(zip(eval_episodes, eval_start_idx, successes)):
                video_dir = videos_success if success else videos_failure
                video_name = f'ep{ep_idx}_start{start_idx}.mp4'
                video_path = video_dir / video_name
                print(f"Saving video for episode {ep_idx} (success={success}) to {video_path}")
                single_cfg = deepcopy(cfg_combo)
                single_cfg.world.num_envs = 1
                single_world = swm.World(**single_cfg.world, image_shape=(224, 224))
                single_world.set_policy(policy)
                tmp_video_dir = Path("tmp_video")
                tmp_video_dir.mkdir(exist_ok=True)
                single_world.evaluate_from_dataset(
                    dataset,
                    start_steps=[int(start_idx)],
                    goal_offset_steps=cfg_combo.eval.goal_offset_steps,
                    eval_budget=cfg_combo.eval.eval_budget,
                    episodes_idx=[int(ep_idx)],
                    callables=OmegaConf.to_container(single_cfg.eval.get('callables'), resolve=True),
                    video_path=tmp_video_dir,
                )
                src_video = tmp_video_dir / "env_0.mp4"
                if src_video.exists():
                    shutil.move(str(src_video), str(video_path))
                else:
                    print(f"Warning: Expected video {src_video} not found.")
        # Save config/results for this combo
        results_file = results_path / cfg_combo.output.filename
        with results_file.open('a') as f:
            f.write('\n')
            f.write('==== CONFIG ====\n')
            f.write(OmegaConf.to_yaml(cfg_combo))
            f.write('\n')
            f.write('==== RESULTS ====\n')
            f.write(f'metrics: {metrics}\n')
            f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
