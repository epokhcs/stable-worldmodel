"""Upload the Glitched Hue TwoRoom dataset to Hugging Face Datasets.

Creates or updates a public dataset repository, uploads the H5 file, and
uploads a generated dataset card (README.md) describing data generation.
It can also upload only PNG/MP4 preview media assets when requested.

Usage:
  python scripts/data/push_glitched_hue_to_hf.py

Optional overrides:
  python scripts/data/push_glitched_hue_to_hf.py \
    --repo-id robomotic/causality-two-room \
    --dataset-path ~/.stable_worldmodel/glitched_hue_tworoom.h5
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Upload glitched TwoRoom H5 dataset to Hugging Face'
    )
    parser.add_argument(
        '--repo-id',
        default='robomotic/causality-two-room',
        help='HF dataset repository id (<user>/<name>)',
    )
    parser.add_argument(
        '--dataset-path',
        default='~/.stable_worldmodel/glitched_hue_tworoom.h5',
        help='Path to local H5 dataset file',
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file containing HF token',
    )
    parser.add_argument(
        '--github-repo-url',
        default=None,
        help='GitHub repository URL to include in dataset README (auto-detected if omitted)',
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create/update dataset as private (default is public)',
    )
    parser.add_argument(
        '--readme-only',
        action='store_true',
        help='Only upload the generated README.md without re-uploading the H5 file',
    )
    parser.add_argument(
        '--media-only',
        action='store_true',
        help='Only upload PNG/MP4 preview files and skip the H5 + README upload',
    )
    parser.add_argument(
        '--media-dir',
        type=Path,
        default=None,
        help='Directory containing PNG/MP4 preview files (default: dataset directory)',
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


def _collect_media_files(media_dir: Path) -> list[Path]:
    """Collect PNG/MP4 media files from a directory recursively."""
    patterns = ('*.png', '*.mp4')
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(media_dir.rglob(pattern)))
    return [path for path in files if path.is_file()]


def _build_readme(
    repo_id: str,
    dataset_path: Path,
    github_url: str,
    created_at: str,
) -> str:
    size_gb = dataset_path.stat().st_size / (1024**3)

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
python scripts/data/collect_glitched_hue.py num_traj=10000 seed=3072
```

Generation details:
- Total episodes: 10,000
- Pass 1: 5,000 episodes with blue background + teleport enabled
- Pass 2: 5,000 episodes with green background + teleport disabled
- Render size: 224x224
- File size: {size_gb:.2f} GB
- Uploaded at (UTC): {created_at}

## Repository

- GitHub: {github_url}

## Notes

The second background hue is green (not red) to preserve strong contrast with
the red agent in pixel observations.
"""


def main() -> None:
    args = _parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            'Missing dependency: huggingface_hub. Install with:\n'
            '  /mnt/sda/stable-worldmodel-causality/.venv/bin/python -m pip install huggingface_hub'
        ) from exc

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not args.media_only and not dataset_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    env_file = Path(args.env_file).expanduser().resolve()
    token = _load_hf_token(env_file)
    github_url = args.github_repo_url or _get_git_remote_url() or 'https://github.com/galilai-group/stable-worldmodel'
    created_at = dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M:%S')

    api = HfApi(token=token)

    # Create repo if needed. exist_ok=True keeps this idempotent.
    api.create_repo(
        repo_id=args.repo_id,
        repo_type='dataset',
        private=bool(args.private),
        exist_ok=True,
    )

    # Ensure visibility matches requested mode (public by default).
    # huggingface_hub versions differ on this API; prefer update_repo_settings.
    api.update_repo_settings(
        repo_id=args.repo_id,
        repo_type='dataset',
        private=bool(args.private),
    )

    if args.media_only:
        media_dir = (
            args.media_dir.expanduser().resolve()
            if args.media_dir is not None
            else dataset_path.parent
        )
        media_files = _collect_media_files(media_dir)
        if not media_files:
            raise FileNotFoundError(
                f'No PNG/MP4 files found under: {media_dir}'
            )

        uploaded_paths: list[str] = []
        for media_file in media_files:
            path_in_repo = f'media/{media_file.name}'
            api.upload_file(
                path_or_fileobj=str(media_file),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type='dataset',
                commit_message='Upload preview media assets',
            )
            uploaded_paths.append(path_in_repo)

        visibility = 'private' if args.private else 'public'
        print('Upload complete')
        print(f'- Repo: https://huggingface.co/datasets/{args.repo_id}')
        print(f'- Visibility: {visibility}')
        print(f'- Media directory: {media_dir}')
        print(f'- Uploaded media files: {len(uploaded_paths)}')
        for repo_path in uploaded_paths:
            print(f'  - {repo_path}')
        return

    readme = _build_readme(
        repo_id=args.repo_id,
        dataset_path=dataset_path,
        github_url=github_url,
        created_at=created_at,
    )

    with tempfile.TemporaryDirectory(prefix='hf_dataset_card_') as tmp_dir:
        readme_path = Path(tmp_dir) / 'README.md'
        readme_path.write_text(readme, encoding='utf-8')

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

    visibility = 'private' if args.private else 'public'
    print('Upload complete')
    print(f'- Repo: https://huggingface.co/datasets/{args.repo_id}')
    print(f'- Visibility: {visibility}')
    if args.readme_only:
        print('- Dataset file upload: skipped (--readme-only)')
    else:
        print(f'- Dataset file: {dataset_path}')
    print(f'- README includes GitHub link: {github_url}')


if __name__ == '__main__':
    main()
