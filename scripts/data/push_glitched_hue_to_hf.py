"""Upload Glitched Hue TwoRoom artifacts to Hugging Face.

By default this uploads the trained LeWM checkpoints to a Hugging Face *model*
repository and generates a full model card with per-epoch metrics extracted
from the local training logs. It can also still upload the HDF5 dataset to a
Hugging Face *dataset* repository when requested.

Examples
--------
Upload the trained checkpoints + model card (default):

    python scripts/data/push_glitched_hue_to_hf.py

Upload to a specific model repo:

    python scripts/data/push_glitched_hue_to_hf.py \
        --repo-type model \
        --repo-id robomotic/causality-two-rooms

Upload the H5 dataset instead:

    python scripts/data/push_glitched_hue_to_hf.py \
        --repo-type dataset \
        --repo-id robomotic/causality-two-room
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MODEL_REPO = 'robomotic/causality-two-rooms'
DEFAULT_DATASET_REPO = 'robomotic/causality-two-room'
DEFAULT_CACHE_DIR = Path('~/.stable_worldmodel').expanduser()
DEFAULT_DATASET_PATH = DEFAULT_CACHE_DIR / 'glitched_hue_tworoom.h5'
DEFAULT_LOG_FILES = [
    Path('/tmp/lewm_glitched_hue_5ep_diag.log'),
    Path('/tmp/lewm_glitched_hue_resume.log'),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Upload Glitched Hue TwoRoom model or dataset artifacts to Hugging Face'
    )
    parser.add_argument(
        '--repo-type',
        choices=('model', 'dataset'),
        default='model',
        help='Whether to upload the trained model checkpoints or the H5 dataset',
    )
    parser.add_argument(
        '--repo-id',
        default=None,
        help='HF repository id (<user>/<name>). Defaults depend on --repo-type.',
    )
    parser.add_argument(
        '--cache-dir',
        default=str(DEFAULT_CACHE_DIR),
        help='Directory containing local checkpoints/config files',
    )
    parser.add_argument(
        '--dataset-path',
        default=str(DEFAULT_DATASET_PATH),
        help='Path to the local H5 dataset file',
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file containing HF token',
    )
    parser.add_argument(
        '--github-repo-url',
        default=None,
        help='GitHub repository URL to include in the generated card (auto-detected if omitted)',
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create/update the repo as private (public by default)',
    )
    parser.add_argument(
        '--readme-only',
        action='store_true',
        help='Upload only the generated README/metrics files and skip large artifacts',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate the card and metrics locally without uploading anything',
    )
    parser.add_argument(
        '--log-file',
        action='append',
        default=[],
        help='Training log file(s) used to extract per-epoch metrics; can be passed multiple times',
    )
    return parser.parse_args()


def _load_hf_token(env_file: Path) -> str:
    candidates = ('HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HF_HUB_TOKEN')

    for key in candidates:
        value = os.getenv(key)
        if value:
            return value.strip().strip('"').strip("'")

    if env_file.exists():
        pattern = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$')
        for line in env_file.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            match = pattern.match(stripped)
            if not match:
                continue
            key, raw_value = match.groups()
            if key not in candidates:
                continue
            value = raw_value.strip().strip('"').strip("'")
            if value:
                return value

    raise RuntimeError(
        'No Hugging Face token found. Set HF_TOKEN/HUGGINGFACE_TOKEN/HF_HUB_TOKEN '
        f'in environment or in {env_file}.'
    )


def _get_git_remote_url() -> str | None:
    try:
        out = subprocess.check_output(
            ['git', 'config', '--get', 'remote.origin.url'],
            text=True,
        ).strip()
    except Exception:
        return None

    if not out:
        return None

    if out.startswith('git@github.com:'):
        slug = out.split(':', 1)[1]
        if slug.endswith('.git'):
            slug = slug[:-4]
        return f'https://github.com/{slug}'

    if out.startswith('https://github.com/'):
        return out[:-4] if out.endswith('.git') else out

    return out


def _human_size(num_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f'{size:.2f} {unit}'
        size /= 1024
    return f'{num_bytes} B'


def _dataset_summary(dataset_path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        'path': str(dataset_path),
        'size_bytes': dataset_path.stat().st_size,
    }
    try:
        import h5py

        with h5py.File(dataset_path, 'r') as handle:
            summary['episodes'] = int(handle['ep_len'].shape[0])
            summary['frames'] = int(handle['pixels'].shape[0])
            summary['pixels_shape'] = tuple(handle['pixels'].shape)
            if 'teleported' in handle:
                summary['teleport_events'] = int(handle['teleported'][:].sum())
    except Exception:
        pass
    return summary


def _epoch_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r'epoch_(\d+)_object\.ckpt$', path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == 'lewm_weights.ckpt':
        return (10**9, path.name)
    return (10**8, path.name)


def _find_checkpoint_files(cache_dir: Path) -> list[Path]:
    checkpoints = sorted(
        cache_dir.glob('lewm_epoch_*_object.ckpt'),
        key=_epoch_sort_key,
    )
    weights = cache_dir / 'lewm_weights.ckpt'
    if weights.exists():
        checkpoints.append(weights)
    return [path.resolve() for path in checkpoints if path.is_file()]


def _find_log_files(paths: list[str]) -> list[Path]:
    if paths:
        candidates = [Path(path).expanduser().resolve() for path in paths]
    else:
        candidates = DEFAULT_LOG_FILES
    return [path for path in candidates if path.exists() and path.is_file()]


def _parse_epoch_metrics(log_paths: list[Path]) -> list[dict[str, object]]:
    pattern = re.compile(
        r'Epoch\s+(\d+)/(\d+)\s+end\s+\|\s+global_step=(\d+)\s+metrics=(\{.*\})'
    )
    results: dict[int, dict[str, object]] = {}

    for path in sorted(log_paths, key=lambda item: (item.stat().st_mtime, str(item))):
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            match = pattern.search(line)
            if not match:
                continue

            epoch = int(match.group(1))
            total_epochs = int(match.group(2))
            global_step = int(match.group(3))
            metrics = ast.literal_eval(match.group(4))
            results[epoch] = {
                'epoch': epoch,
                'total_epochs': total_epochs,
                'global_step': global_step,
                **metrics,
            }

    return [results[key] for key in sorted(results)]


def _write_metrics_files(metrics_rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'epoch_metrics.json'
    csv_path = output_dir / 'epoch_metrics.csv'

    json_path.write_text(json.dumps(metrics_rows, indent=2), encoding='utf-8')

    fieldnames: list[str] = ['epoch', 'total_epochs', 'global_step']
    seen = set(fieldnames)
    for row in metrics_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    return json_path, csv_path


def _metrics_markdown_table(metrics_rows: list[dict[str, object]]) -> str:
    if not metrics_rows:
        return 'No epoch-end metrics could be extracted from the local training logs.'

    lines = [
        '| Epoch | Global step | fit/loss | fit/pred_loss | fit/sigreg_loss | validate/loss | validate/pred_loss | validate/sigreg_loss |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in metrics_rows:
        lines.append(
            '| {epoch} | {global_step} | {fit_loss:.6f} | {fit_pred:.6f} | {fit_sigreg:.6f} | {val_loss:.6f} | {val_pred:.6f} | {val_sigreg:.6f} |'.format(
                epoch=int(row.get('epoch', 0)),
                global_step=int(row.get('global_step', 0)),
                fit_loss=float(row.get('fit/loss', float('nan'))),
                fit_pred=float(row.get('fit/pred_loss', float('nan'))),
                fit_sigreg=float(row.get('fit/sigreg_loss', float('nan'))),
                val_loss=float(row.get('validate/loss', row.get('validate/loss_epoch', float('nan')))),
                val_pred=float(row.get('validate/pred_loss', row.get('validate/pred_loss_epoch', float('nan')))),
                val_sigreg=float(row.get('validate/sigreg_loss', row.get('validate/sigreg_loss_epoch', float('nan')))),
            )
        )
    return '\n'.join(lines)


def _artifact_markdown_table(checkpoints: list[Path], config_path: Path | None) -> str:
    lines = [
        '| File | Purpose | Size |',
        '|---|---|---:|',
    ]
    for checkpoint in checkpoints:
        purpose = 'Full Lightning trainer checkpoint' if checkpoint.name == 'lewm_weights.ckpt' else 'Serialized model object checkpoint'
        lines.append(
            f'| `checkpoints/{checkpoint.name}` | {purpose} | {_human_size(checkpoint.stat().st_size)} |'
        )
    if config_path and config_path.exists():
        lines.append(
            f'| `config.yaml` | Hydra config used for the run | {_human_size(config_path.stat().st_size)} |'
        )
    lines.append('| `metrics/epoch_metrics.json` | Raw epoch metrics extracted from local logs | small |')
    lines.append('| `metrics/epoch_metrics.csv` | Tabular epoch metrics for spreadsheets / plotting | small |')
    return '\n'.join(lines)


def _build_model_card(
    repo_id: str,
    github_url: str,
    created_at: str,
    checkpoints: list[Path],
    dataset_info: dict[str, object] | None,
    metrics_rows: list[dict[str, object]],
    config_path: Path | None,
) -> str:
    dataset_bits: list[str] = []
    if dataset_info:
        dataset_bits.append(f"- Dataset path: `{dataset_info.get('path')}`")
        dataset_bits.append(f"- Dataset size: {_human_size(int(dataset_info.get('size_bytes', 0)))}")
        if 'episodes' in dataset_info:
            dataset_bits.append(f"- Episodes: {dataset_info['episodes']:,}")
        if 'frames' in dataset_info:
            dataset_bits.append(f"- Frames: {dataset_info['frames']:,}")
        if 'pixels_shape' in dataset_info:
            dataset_bits.append(f"- Pixel tensor shape: `{dataset_info['pixels_shape']}`")
        if 'teleport_events' in dataset_info:
            dataset_bits.append(f"- Teleport events: {dataset_info['teleport_events']:,}")

    dataset_section = '\n'.join(dataset_bits) if dataset_bits else '- Dataset summary unavailable from local cache.'

    config_lines = [
        '| Parameter | Value |',
        '|---|---|',
        '| `trainer.max_epochs` | `5` |',
        '| `trainer.accelerator` | `gpu` |',
        '| `trainer.precision` | `bf16` |',
        '| `loader.batch_size` | `128` |',
        '| `loader.num_workers` | `1` for the resumed run |',
        '| `optimizer.lr` | `5e-5` |',
        '| `wm.history_size` | `3` |',
        '| `wm.num_preds` | `1` |',
        '| `wm.embed_dim` | `192` |',
        '| `loss.sigreg.weight` | `0.09` |',
        '| `data.dataset.frameskip` | `5` |',
    ]
    if config_path and config_path.exists():
        config_lines.append(f'| `config.yaml` | included in the repo root |')

    return f"""---
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
- robotics
- reinforcement-learning
- world-model
- causal-representation-learning
- stable-worldmodel
---

# {repo_id}

LeWM checkpoints trained on the confounded **Glitched Hue TwoRoom** dataset for
causal world-model experiments. The goal is to test whether the model learns the
true teleport mechanism or the spurious background-hue correlation.

## Model description

- **Architecture:** LeWM / JEPA-style world model with an autoregressive predictor
- **Domain:** `swm/GlitchedHueTwoRoom-v1`
- **Framework:** PyTorch + Lightning
- **Repository:** {github_url}
- **Upload generated (UTC):** {created_at}

## Training data

{dataset_section}

The dataset was collected with:

```bash
python scripts/data/collect_glitched_hue.py \
    num_traj=10000 \
    seed=3072 \
    world.num_envs=10
```

## Training procedure

The checkpoints in this repo come from the 5-epoch LeWM training run used in the
causality experiment. The run completed successfully after resuming from the last
full trainer checkpoint.

Command family:

```bash
python scripts/train/lewm.py \
    data=glitched_hue_tworoom \
    trainer.max_epochs=5 \
    num_workers=1 \
    loader.num_workers=1 \
    loader.persistent_workers=False
```

### Key hyperparameters

{chr(10).join(config_lines)}

## Epoch metrics (logged to W&B / Lightning)

The table below summarizes the epoch-end losses extracted from the local training
logs. The raw values are also included as `metrics/epoch_metrics.json` and
`metrics/epoch_metrics.csv`.

{_metrics_markdown_table(metrics_rows)}

## Files in this repo

{_artifact_markdown_table(checkpoints, config_path)}

## How to use

Load a serialized model-object checkpoint:

```python
import torch

model = torch.load('checkpoints/lewm_epoch_5_object.ckpt', map_location='cpu')
model.eval()
```

Load the full Lightning trainer checkpoint:

```python
import torch

checkpoint = torch.load('checkpoints/lewm_weights.ckpt', map_location='cpu')
print(checkpoint.keys())
```

## Intended uses

- Reproducing the causal disentanglement experiment in `research/runme.md`
- Running the Step 3 causal AAP analysis with `research/glitched_hue_experiment.py`
- Comparing epoch-wise world-model checkpoints during training

## Limitations

- These checkpoints are research artifacts, not production control policies.
- Performance is specific to the Glitched Hue TwoRoom environment and the
  confounded blue/green data collection procedure.
- The object checkpoints are convenient for inspection, while the full trainer
  checkpoint is the correct file for resuming optimization.
"""


def _build_dataset_card(
    repo_id: str,
    dataset_path: Path,
    github_url: str,
    created_at: str,
    dataset_info: dict[str, object],
) -> str:
    size_gb = dataset_path.stat().st_size / (1024**3)
    episodes = dataset_info.get('episodes', 'unknown')
    frames = dataset_info.get('frames', 'unknown')
    pixels_shape = dataset_info.get('pixels_shape', 'unknown')
    teleports = dataset_info.get('teleport_events', 'unknown')

    return f"""---
license: mit
task_categories:
- reinforcement-learning
- robotics
language:
- en
pretty_name: Causality Two Room (Glitched Hue)
size_categories:
- 10K<n<100K
---

# {repo_id}

This dataset contains trajectories collected in the `swm/GlitchedHueTwoRoom-v1`
environment for causal world-model experiments.

## What is inside

- `glitched_hue_tworoom.h5`: HDF5 dataset with trajectories and rendered frames.
- Includes observations, actions, rewards, episode indexing, variation values,
  and pixel renderings.

## How it was generated

Collection script:
- `scripts/data/collect_glitched_hue.py`

Configuration:
- `scripts/data/config/glitched_hue.yaml`

Command used:

```bash
python scripts/data/collect_glitched_hue.py num_traj=10000 seed=3072 world.num_envs=10
```

Generation details:
- Total episodes: {episodes}
- Total frames: {frames}
- Pixel shape: `{pixels_shape}`
- Teleport events: {teleports}
- File size: {size_gb:.2f} GB
- Uploaded at (UTC): {created_at}

## Repository

- GitHub: {github_url}

## Notes

The second background hue is green (not red) to preserve strong contrast with
the red agent in pixel observations.
"""


def _safe_update_repo_settings(api, repo_id: str, repo_type: str, private: bool) -> None:
    try:
        api.update_repo_settings(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
        )
    except Exception:
        # Older huggingface_hub versions may not expose this method.
        pass


def _upload_model_artifacts(api, args: argparse.Namespace, github_url: str, created_at: str) -> None:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    config_path = cache_dir / 'config.yaml'
    checkpoints = _find_checkpoint_files(cache_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f'No model checkpoints found under {cache_dir}. Expected files like '
            '`lewm_epoch_5_object.ckpt` or `lewm_weights.ckpt`.'
        )

    log_files = _find_log_files(args.log_file)
    metrics_rows = _parse_epoch_metrics(log_files)
    dataset_info = _dataset_summary(dataset_path) if dataset_path.exists() else None

    with tempfile.TemporaryDirectory(prefix='hf_model_card_') as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        metrics_dir = tmp_dir_path / 'metrics'
        json_path, csv_path = _write_metrics_files(metrics_rows, metrics_dir)
        readme_path = tmp_dir_path / 'README.md'
        readme_path.write_text(
            _build_model_card(
                repo_id=args.repo_id,
                github_url=github_url,
                created_at=created_at,
                checkpoints=checkpoints,
                dataset_info=dataset_info,
                metrics_rows=metrics_rows,
                config_path=config_path if config_path.exists() else None,
            ),
            encoding='utf-8',
        )

        print(f'Prepared model card: {readme_path}')
        print(f'Prepared metrics JSON: {json_path}')
        print(f'Prepared metrics CSV: {csv_path}')
        print(f'Checkpoints discovered: {len(checkpoints)}')
        for checkpoint in checkpoints:
            print(f'  - {checkpoint.name} ({_human_size(checkpoint.stat().st_size)})')

        if args.dry_run:
            print('\nDry run only; nothing was uploaded.')
            return

        if not args.readme_only:
            for checkpoint in checkpoints:
                api.upload_file(
                    path_or_fileobj=str(checkpoint),
                    path_in_repo=f'checkpoints/{checkpoint.name}',
                    repo_id=args.repo_id,
                    repo_type='model',
                    commit_message=f'Add {checkpoint.name}',
                )
            if config_path.exists():
                api.upload_file(
                    path_or_fileobj=str(config_path),
                    path_in_repo='config.yaml',
                    repo_id=args.repo_id,
                    repo_type='model',
                    commit_message='Add training config',
                )

        for artifact_path, repo_path in (
            (json_path, 'metrics/epoch_metrics.json'),
            (csv_path, 'metrics/epoch_metrics.csv'),
            (readme_path, 'README.md'),
        ):
            api.upload_file(
                path_or_fileobj=str(artifact_path),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type='model',
                commit_message=f'Update {repo_path}',
            )

    print('Upload complete')
    print(f'- Repo: https://huggingface.co/{args.repo_id}')
    print(f'- Visibility: {"private" if args.private else "public"}')
    print(f'- Metrics rows uploaded: {len(metrics_rows)}')
    if args.readme_only:
        print('- Large artifacts skipped (--readme-only)')


def _upload_dataset_artifacts(api, args: argparse.Namespace, github_url: str, created_at: str) -> None:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    dataset_info = _dataset_summary(dataset_path)
    readme = _build_dataset_card(
        repo_id=args.repo_id,
        dataset_path=dataset_path,
        github_url=github_url,
        created_at=created_at,
        dataset_info=dataset_info,
    )

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    stats_path = cache_dir / 'episode_stats.md'
    _VIDEO_LABELS = ('blue_success', 'blue_failure', 'green_success', 'green_failure')
    videos_dir = Path(os.getenv('VIDEOS_DIR', cache_dir / 'videos'))
    video_paths = {
        label: videos_dir / f'{label}.mp4'
        for label in _VIDEO_LABELS
        if (videos_dir / f'{label}.mp4').exists()
    }

    with tempfile.TemporaryDirectory(prefix='hf_dataset_card_') as tmp_dir:
        readme_path = Path(tmp_dir) / 'README.md'
        readme_path.write_text(readme, encoding='utf-8')

        if args.dry_run:
            print(f'Prepared dataset card: {readme_path}')
            if stats_path.exists():
                print(f'Would upload stats report: {stats_path}')
            else:
                print(f'Stats report not found (skipping): {stats_path}')
            for label, vp in video_paths.items():
                print(f'Would upload video: videos/{label}.mp4')
            if not video_paths:
                print(f'No sample videos found in {videos_dir}')
            print('Dry run only; nothing was uploaded.')
            return

        if not args.readme_only:
            api.upload_file(
                path_or_fileobj=str(dataset_path),
                path_in_repo=dataset_path.name,
                repo_id=args.repo_id,
                repo_type='dataset',
                commit_message='Add glitched TwoRoom H5 dataset',
            )

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo='README.md',
            repo_id=args.repo_id,
            repo_type='dataset',
            commit_message='Refresh dataset card metadata and generation details',
        )

        if stats_path.exists():
            api.upload_file(
                path_or_fileobj=str(stats_path),
                path_in_repo='episode_stats.md',
                repo_id=args.repo_id,
                repo_type='dataset',
                commit_message='Add episode statistics report',
            )
            print(f'- Stats report uploaded: episode_stats.md')
        else:
            print(f'- Stats report not found, skipped: {stats_path}')

        for label, vp in video_paths.items():
            api.upload_file(
                path_or_fileobj=str(vp),
                path_in_repo=f'videos/{label}.mp4',
                repo_id=args.repo_id,
                repo_type='dataset',
                commit_message=f'Add sample video: {label}',
            )
            print(f'- Video uploaded: videos/{label}.mp4')
        if not video_paths:
            print(f'- No sample videos found in {videos_dir}, skipped')

    print('Upload complete')
    print(f'- Repo: https://huggingface.co/datasets/{args.repo_id}')
    print(f'- Visibility: {"private" if args.private else "public"}')
    if args.readme_only:
        print('- Dataset file upload: skipped (--readme-only)')
    else:
        print(f'- Dataset file: {dataset_path}')


def main() -> None:
    args = _parse_args()
    if args.repo_id is None:
        args.repo_id = (
            DEFAULT_MODEL_REPO if args.repo_type == 'model' else DEFAULT_DATASET_REPO
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            'Missing dependency: huggingface_hub. Install with:\n'
            '  /mnt/sda/stable-worldmodel-causality/.venv/bin/python -m pip install huggingface_hub'
        ) from exc

    env_file = Path(args.env_file).expanduser().resolve()
    token = _load_hf_token(env_file)
    github_url = (
        args.github_repo_url
        or _get_git_remote_url()
        or 'https://github.com/epokhcs/stable-worldmodel'
    )
    created_at = dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M:%S')

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=bool(args.private),
        exist_ok=True,
    )
    _safe_update_repo_settings(api, args.repo_id, args.repo_type, bool(args.private))

    if args.repo_type == 'model':
        _upload_model_artifacts(api, args, github_url, created_at)
    else:
        _upload_dataset_artifacts(api, args, github_url, created_at)


if __name__ == '__main__':
    main()
