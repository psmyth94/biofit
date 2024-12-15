# Copyright 2024 Patrick Smyth and Hugging Face authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import multiprocessing.pool
import os
import queue
import random
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from queue import Empty
from typing import Callable, Iterable, Set, TypeVar, Union

import multiprocess
import multiprocess.pool
import numpy as np
from biocore.utils.import_util import (
    is_datasets_available,
    is_polars_available,
    is_tf_available,
    is_torch_available,
)

Y = TypeVar("Y")


def is_temporal(val):
    return isinstance(val, (datetime, date, time, timedelta))


def is_decimal(val):
    return isinstance(val, Decimal)


def as_py(val):
    # return as int64
    if isinstance(val, datetime):
        return val.timestamp()
    elif isinstance(val, date):
        return val.toordinal()
    elif isinstance(val, time):
        return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
    elif isinstance(val, timedelta):
        return val.total_seconds()
    elif isinstance(val, Decimal):
        return float(val)
    elif isinstance(val, np.float16):
        return float(val)
    elif isinstance(val, bytes):
        return f"base64:{val.hex()}"
    elif isinstance(val, dict):
        return {k: as_py(v) for k, v in val.items()}
    return val


def enable_full_determinism(seed: int, warn_only: bool = False):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # set seed first
    set_seed(seed)

    if is_torch_available():
        import torch

        # Enable PyTorch deterministic mode. This potentially requires either the environment
        # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
        # depending on the CUDA version, so we set them both here
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if is_tf_available():
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    if is_polars_available():
        from polars import set_random_seed

        set_random_seed(seed)

    def is_torch_npu_available():
        try:
            import torch

            return torch.npu.is_available()
        except ImportError:
            return False

    if is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


# This function is a copy of the one in datasets.utils.py_util.py, Copyright 2024
# Hugging Face authors. Licensed under the Apache 2.0 license. See the license file for
# details at https://www.apache.org/licenses/LICENSE-2.0
def _get_pool_pid(
    pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool]
) -> Set[int]:
    return {f.pid for f in pool._pool}


# This function is a copy of the one in datasets.utils.py_util.py, Copyright 2024
# Hugging Face authors. Licensed under the Apache 2.0 license. See the license file for
# details at https://www.apache.org/licenses/LICENSE-2.0
def _write_generator_to_queue(
    queue: queue.Queue, func: Callable[..., Iterable[Y]], kwargs: dict
) -> int:
    for i, result in enumerate(func(**kwargs)):
        queue.put(result)
    return i


# This function is a copy of the one in datasets.utils.py_util.py, Copyright 2024
# Hugging Face authors. Licensed under the Apache 2.0 license. See the license file for
# details at https://www.apache.org/licenses/LICENSE-2.0
def iflatmap_unordered(
    pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool],
    func: Callable[..., Iterable[Y]],
    *,
    kwargs_iterable: Iterable[dict],
) -> Iterable[Y]:
    """
    Similar to `itertools.chain.from_iterable(map(func, iterable))` but with a potentially more efficient implementation
    that doesn't require storing all the intermediate results in memory.

    Args:
        func (Callable): The function to apply to each element of the input iterable.
        iterable (Iterable): The input iterable.

    Returns:
        Iterator: The flattened iterable.
    """
    if is_datasets_available():
        from datasets.utils.py_utils import iflatmap_unordered

        return iflatmap_unordered(pool, func, kwargs_iterable=kwargs_iterable)
    else:
        initial_pool_pid = _get_pool_pid(pool)
        pool_changed = False
        manager_cls = (
            multiprocessing.Manager
            if isinstance(pool, multiprocessing.pool.Pool)
            else multiprocess.Manager
        )
        with manager_cls() as manager:
            queue = manager.Queue()
            async_results = [
                pool.apply_async(_write_generator_to_queue, (queue, func, kwargs))
                for kwargs in kwargs_iterable
            ]
            try:
                while True:
                    try:
                        yield queue.get(timeout=0.05)
                    except Empty:
                        if (
                            all(async_result.ready() for async_result in async_results)
                            and queue.empty()
                        ):
                            break
                    if _get_pool_pid(pool) != initial_pool_pid:
                        pool_changed = True
                        # One of the subprocesses has died. We should not wait forever.
                        raise RuntimeError(
                            "One of the subprocesses has abruptly died during map operation."
                            "To debug the error, disable multiprocessing."
                        )
            finally:
                if not pool_changed:
                    # we get the result in case there's an error to raise
                    [async_result.get(timeout=0.05) for async_result in async_results]
