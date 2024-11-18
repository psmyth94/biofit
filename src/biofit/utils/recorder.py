from typing import List, Optional, Union, Callable
import sys
import inspect
import importlib
from functools import wraps
from pathlib import Path


from .. import config
from .file_utils import is_file_name, PathLike
from . import logging

logger = logging.get_logger(__name__)


def load_module_or_class(full_path):
    if full_path is None:
        return None
    try:
        module_path, entity_name = full_path.rsplit(".", 1)
        module = (
            importlib.import_module(module_path)
            if module_path not in sys.modules
            else sys.modules[module_path]
        )
    except ImportError:
        # Handle the case where the module can't be imported
        logger.error(f"Failed to import module: {module_path}")
        return None

    try:
        entity = (
            getattr(module, entity_name)
            if entity_name not in sys.modules
            else sys.modules[entity_name]
        )
    except AttributeError:
        # Handle the case where the entity isn't found in the module
        logger.error(f"{entity_name} not found in module: {module_path}")
        return None

    return entity


def load_method(full_path, method_name):
    entity = load_module_or_class(full_path)
    if entity:
        return getattr(entity, method_name)
    return None


def _get_cache_dir(
    cache_files: Optional[List[dict]] = None, cache_file_name: Optional[PathLike] = None
) -> Optional[Path]:
    # check if cache_file_name is a path
    cache_dir = None
    if cache_file_name is not None:
        if isinstance(cache_file_name, PathLike):
            cache_file_name = Path(cache_file_name)
            if "/" in cache_file_name.as_posix():
                cache_dir = cache_file_name.resolve().parent

    if not cache_dir and cache_files:
        cache_dir = Path(cache_files[0]["filename"]).resolve().parent

    return cache_dir


def _get_cache_info(
    cache_files: Optional[List[dict]] = None,
    cache_dir: Optional[Path] = None,
    cache_file_name: Optional[Union[str, Path]] = None,
    file_ext=".arrow",
):
    if cache_file_name:
        if is_file_name(cache_file_name):
            cache_dir = cache_dir or Path.cwd()
            cache_file_name = Path(cache_dir) / Path(cache_file_name).with_suffix(
                file_ext
            )
        else:
            cache_file_name = Path(cache_file_name).with_suffix(file_ext)

    cache_dir = _get_cache_dir(cache_files, cache_file_name)
    return cache_file_name, cache_dir


UNRECORDED_METHODS = ["train_test_split"]
_RECORDER_ACTIVE = False


def pre_recording(func, *args, **kwargs):
    new_fingerprint = kwargs.get("new_fingerprint", kwargs.get("fingerprint", None))

    signature = inspect.signature(func)
    cache_file_name = kwargs.get("cache_file_name", None)
    if not cache_file_name and "cache_file_name" in signature.parameters:
        arg_pos = list(signature.parameters).index("cache_file_name")
        if len(args) > arg_pos:
            cache_file_name = args[arg_pos]
    cache_dir = None
    self = args[0] if args else kwargs.get("self", None)
    if getattr(self, "cache_files", None):
        cache_file_name, cache_dir = _get_cache_info(
            self.cache_files, cache_dir, cache_file_name
        )
    if isinstance(cache_dir, Path):
        cache_dir = cache_dir.resolve().as_posix()
    if isinstance(cache_file_name, Path):
        cache_file_name = cache_file_name.resolve().as_posix()

    return {
        "cache_file_name": cache_file_name,
        "cache_dir": cache_dir,
        "new_fingerprint": new_fingerprint,
    }


def toggle_recorder():
    global _RECORDER_ACTIVE
    _RECORDER_ACTIVE = not _RECORDER_ACTIVE


def record_step(
    replay_func: Optional[str] = None,
    pre_recording: Optional[Callable] = pre_recording,
    post_recording: Optional[Callable] = None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.RECORDER_ENABLED or _RECORDER_ACTIVE:
                return func(*args, **kwargs)

            toggle_recorder()
            extra_info = {}
            if pre_recording:
                extra_info = pre_recording(func, *args, **kwargs)
            out = func(*args, **kwargs)
            if post_recording:
                out = post_recording(
                    out,
                    func.__name__,
                    args,
                    kwargs,
                    replay_func=replay_func,
                    info=extra_info,
                )

            toggle_recorder()

            return out

        return wrapper

    return decorator
