
import subprocess
from pathlib import Path
import modal

app = modal.App("stable-worldmodel-causality")

# The cache directory for datasets
CACHE_DIR = "/root/.stable_worldmodel"
# The directory for videos
VIDEOS_DIR = "/workspace/videos"

# Create shared persistent volumes automatically if they don't exist
volume = modal.Volume.from_name("swm-cache", create_if_missing=True)
videos_volume = modal.Volume.from_name("swm-videos", create_if_missing=True)

# Required secrets for logging and models
# Ensure these secrets exist in your Modal workspace via `modal secret create`
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

# Common environment variables for all jobs
COMMON_ENV = {
    "STABLEWM_HOME": CACHE_DIR,
    "MUJOCO_GL": "egl",
}

# CPU-only image — data collection runs physics sims, no GPU needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1",
        "libegl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "ffmpeg",
    )
    # CPU-only torch build (~700 MB vs ~2 GB for CUDA)
    .pip_install(
        "torch", "torchvision",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    # Core package runtime deps
    .pip_install(
        "numpy", "loguru", "tabulate", "einops",
        "h5py", "hdf5plugin", "tqdm", "gdown", "typer", "rich",
    )
    # Environment extras (simulation + rendering)
    .pip_install(
        "pygame", "pymunk", "shapely",
        "ogbench", "minigrid", "gymnasium[all]",
        "opencv-python", "stable_baselines3",
    )
    # Collection script config system + HF upload
    .pip_install("hydra-core", "huggingface_hub")
    .add_local_dir(
        "/mnt/sda/stable-worldmodel-causality",
        remote_path="/workspace",
        copy=True,
        ignore=[".venv", "venv", ".git", "outputs", "wandb", "__pycache__", "modal"],
    )
    .workdir("/workspace")
    .run_commands("pip install --no-deps -e .")
)


# Only dataset collection function is kept
@app.function(
    image=image,
    volumes={CACHE_DIR: volume, VIDEOS_DIR: videos_volume},
    secrets=[hf_secret],
    env=COMMON_ENV,
    timeout=86400,
)
def collect_dataset(
    num_traj: int = 20000,
    seed: int = 3072,
    num_envs: int = 10,
    dataset_repo: str = "robomotic/causality-two-room-modal"
):
    print(f"Starting dataset collection: {num_traj} trajectories...")
    cmd = [
        "python", "scripts/data/collect_glitched_hue.py",
        f"num_traj={num_traj}",
        f"seed={seed}",
        f"world.num_envs={num_envs}"
    ]
    subprocess.run(cmd, check=True)
    volume.commit()
    print("Dataset collected and persisted to volume.")

    print("Computing episode statistics...")
    subprocess.run(["python", "scripts/data/compute_episode_stats.py"], check=True)
    volume.commit()
    print("Statistics written to volume.")

    print(f"Pushing dataset to HF ({dataset_repo})...")
    push_cmd = [
        "python", "scripts/data/push_glitched_hue_to_hf.py",
        "--repo-type", "dataset",
        "--repo-id", dataset_repo
    ]
    subprocess.run(push_cmd, check=True)
    print("Dataset pushed to HF successfully.")

@app.local_entrypoint()
def main(
    collect: bool = False,
    all: bool = False,
    num_traj: int = 20000,
    dataset_repo: str = "robomotic/causality-two-room-modal"
):
    """
    Local entrypoint to orchestrate phases on Modal.
    Usage example: modal run modal/app.py --all
    """
    if all:
        collect = True
        
    if collect:
        collect_dataset.remote(num_traj=num_traj, dataset_repo=dataset_repo)
        