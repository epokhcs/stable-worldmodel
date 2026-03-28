import inspect
import json
import urllib.request
from pathlib import Path
import torch

from stable_worldmodel.wm.module import is_swm_model

from loguru import logger as logging
from tqdm import tqdm

from stable_worldmodel.utils import HF_BASE_URL
from stable_worldmodel.data import get_cache_dir, ensure_dir_exists


class AutoModel:
    """Load a model from a local checkpoint or a HuggingFace repository.

    Supported formats for `name`:

    1. **`.pt` file** — path to a specific checkpoint file.
       A `config.json` must live in the same directory.

        ```python
        model = AutoModel.from_pretrained('my_run/weights_epoch_10.pt', MyModel)
        ```

    2. **Folder** — path to a directory containing exactly one `.pt` file
       and a `config.json`.

        ```python
        model = AutoModel.from_pretrained('my_run/', MyModel)
        ```

    3. **HuggingFace repo** (`<user>/<repo>`) — loaded from the local cache
       if already present, otherwise fetched from HF.

        ```python
        model = AutoModel.from_pretrained('nice-user/my-worldmodel', MyModel)
        ```

    All local paths are resolved relative to `<cache_dir>/checkpoints/`.
    """

    @classmethod
    def from_pretrained(cls, name: str, cache_dir: str = None):
        """Return a loaded model instance from a local checkpoint or HF repo."""
        cache_dir = get_cache_dir(cache_dir, sub_folder='checkpoints')
        ensure_dir_exists(cache_dir)
        checkpoint_path, config = cls._resolve(name, cache_dir)
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        if 'architecture' not in config:
            raise ValueError(
                f"Config for '{name}' is missing 'architecture' field. "
                'Cannot determine which model class to instantiate.'
            )

        model_cls = cls._resolve_model(config['architecture'])
        return model_cls.from_config(config).load_state_dict(state_dict)

    @classmethod
    def from_config(cls, config: dict):
        """Instantiate a model from a config dict."""
        if 'architecture' not in config:
            raise ValueError(
                'Impossible to auto-load model: config missing "architecture" field.'
            )
        model_cls = cls._resolve_model(config['architecture'])
        return model_cls.from_config(config)

    @staticmethod
    def _resolve_model(architecture: str):
        """Import and return the model class specified by *architecture*."""
        module_path, class_name = architecture.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    @staticmethod
    def _resolve(name: str, cache_dir: Path) -> tuple[Path, dict]:
        """Return ``(checkpoint_path, config_dict)`` for *name*.

        Resolution order:
          1. ``<cache_dir>/<name>``  as a ``.pt`` file
          2. ``<cache_dir>/<name>``  as a folder
          3. HuggingFace repo (cached locally under ``<cache_dir>/<user>/<repo>/``)
        """
        local = cache_dir / name

        # format 1: explicit .pt file
        if local.suffix == '.pt':
            if not local.exists():
                raise FileNotFoundError(f'Checkpoint not found: {local}')
            return local, AutoModel._load_config(local.parent)

        # format 2: folder containing a .pt and config.json
        if local.is_dir():
            return AutoModel._resolve_folder(local)

        # format 3: HuggingFace repo (<user>/<repo>)
        if '/' in name:
            return AutoModel._resolve_hf(name, cache_dir)

        raise ValueError(
            f"Cannot resolve '{name}': not a .pt file, a folder, or a HF repo id."
        )

    @staticmethod
    def _resolve_folder(folder: Path) -> tuple[Path, dict]:
        """Load from a folder containing one ``.pt`` file and a ``config.json``."""
        pt_files = list(folder.glob('*.pt'))
        if not pt_files:
            raise FileNotFoundError(f'No .pt file found in {folder}')
        if len(pt_files) > 1:
            raise ValueError(
                f'Ambiguous checkpoint: multiple .pt files in {folder}. '
                'Specify the file directly.'
            )
        logging.info(f'Loading checkpoint from folder {folder}...')
        return pt_files[0], AutoModel._load_config(folder)

    @staticmethod
    def _resolve_hf(repo_id: str, cache_dir: Path) -> tuple[Path, dict]:
        """Resolve a HuggingFace repo id, using a local cache when available.

        Local layout: ``<cache_dir>/models--<user>--<repo>/``
        """
        local_dir = cache_dir / f'models--{repo_id.replace("/", "--")}'

        if local_dir.is_dir():
            logging.info(f'Loading {repo_id} from local cache...')
            return AutoModel._resolve_folder(local_dir)

        logging.info(f'Downloading {repo_id} from HuggingFace...')
        local_dir.mkdir(parents=True, exist_ok=True)
        for filename in ('config.json', 'weights.pt'):
            url = f'{HF_BASE_URL}/{repo_id}/resolve/main/{filename}'
            dest = local_dir / filename
            logging.info(f'Fetching {url}')
            AutoModel._download(url, dest)

        return AutoModel._resolve_folder(local_dir)

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Download *url* to *dest* with a tqdm progress bar."""
        response = urllib.request.urlopen(url)
        total = int(response.headers.get('Content-Length', 0)) or None
        with (
            open(dest, 'wb') as f,
            tqdm(
                total=total, unit='B', unit_scale=True, desc=dest.name
            ) as bar,
        ):
            while chunk := response.read(8192):
                f.write(chunk)
                bar.update(len(chunk))

    @staticmethod
    def _load_config(folder: Path) -> dict:
        config_path = folder / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'config.json not found in {folder}')
        with open(config_path) as f:
            return json.load(f)


def save(
    module: torch.nn.Module,
    run_name: str,
    filename: str = 'weights.pt',
    cache_dir: str = None,
):
    """Save a model checkpoint and config to the cache directory."""
    ...

    cache_dir = get_cache_dir(cache_dir, sub_folder='checkpoints')
    run_dir = cache_dir / run_name
    ensure_dir_exists(run_dir)
    checkpoint_path = run_dir / filename
    torch.save(module.state_dict(), checkpoint_path)

    # save config
    config = dump_config(module)
    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return


def dump_config(module: torch.nn.Module) -> dict:
    """Return a config dict for *module*.

    Resolution order:
    1. ``Model`` protocol — if the module has a ``config`` property (i.e. it
       satisfies :class:`~stable_worldmodel.wm.module.Model`), return it directly.
    2. HuggingFace — if the module has a ``.config.to_dict()``, extract the
       primitive fields from it.
    3. Introspection — walk the ``__init__`` signature and read same-named
       instance attributes; sub-modules are recursed into.
    """

    if is_swm_model(module):
        return module.config

    # HuggingFace models expose a structured config object
    if hasattr(module, 'config') and hasattr(module.config, 'to_dict'):
        return module.config.to_dict()

    logging.warning(
        f'Module {module.__class__.__name__} does not satisfy the Model protocol '
        'and has no HF-style config. Falling back to introspection, which may '
        'miss some fields or include non-primitive values.'
    )

    sig = inspect.signature(module.__init__)
    result = {}
    for name, _ in sig.parameters.items():
        if name in ('self', 'args', 'kwargs'):
            continue
        if not hasattr(module, name):
            continue
        val = getattr(module, name)
        if isinstance(val, torch.nn.Module):
            result[name] = dump_config(val)
        elif isinstance(val, (int, float, str, bool)) or val is None:
            result[name] = val
    return result


__all__ = ['AutoModel', 'dump_config', 'save']
