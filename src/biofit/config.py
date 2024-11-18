import importlib
import importlib.metadata
import importlib.util
import logging
import os
from pathlib import Path

from packaging import version

logger = logging.getLogger(__name__)

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_FALSE_VALUES = {"0", "OFF", "NO", "FALSE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})
ENV_VARS_FALSE_AND_AUTO_VALUES = ENV_VARS_FALSE_VALUES.union({"AUTO"})


DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_BIOFIT_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "biofit")
BIOFIT_CACHE_HOME = os.path.expanduser(
    os.getenv("BIOFIT_HOME", DEFAULT_BIOFIT_CACHE_HOME)
)

DEFAULT_BIOFIT_DATASETS_CACHE = os.path.join(BIOFIT_CACHE_HOME, "datasets")
BIOFIT_DATASETS_CACHE = Path(
    os.getenv("BIOFIT_DATASETS_CACHE", DEFAULT_BIOFIT_DATASETS_CACHE)
)
DEFAULT_BIOFIT_PREPROCESSORS_CACHE = os.path.join(BIOFIT_CACHE_HOME, "processors")
BIOFIT_PROCESSORS_CACHE = Path(
    os.getenv("BIOFIT_PROCESSORS_CACHE", DEFAULT_BIOFIT_PREPROCESSORS_CACHE)
)

DEFAULT_BIOFIT_METRICS_CACHE = os.path.join(BIOFIT_CACHE_HOME, "metrics")
BIOFIT_METRICS_CACHE = Path(
    os.getenv("BIOFIT_METRICS_CACHE", DEFAULT_BIOFIT_METRICS_CACHE)
)

DEFAULT_BIOFIT_MODULES_CACHE = os.path.join(BIOFIT_CACHE_HOME, "modules")
BIOFIT_MODULES_CACHE = Path(
    os.getenv("BIOFIT_MODULES_CACHE", DEFAULT_BIOFIT_MODULES_CACHE)
)

BIOFIT_DYNAMIC_MODULE_NAME = Path(
    os.getenv("BIOFIT_DYNAMIC_MODULE_NAME", "datasets_modules")
)

BIOFIT_CACHE_HOME = os.getenv(
    "BIOFIT_CACHE_HOME",
    os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "biofit"),
)
DEFAULT_BIOFIT_PATCHES_CACHE = os.path.join(BIOFIT_CACHE_HOME, "patches")
BIOFIT_PATCHES_CACHE = Path(
    os.getenv("BIOFIT_PATCHES_CACHE", DEFAULT_BIOFIT_PATCHES_CACHE)
)

DOWNLOADED_PATCHES_DIR = "downloads"
DEFAULT_DOWNLOADED_PATCHES_PATH = os.path.join(
    BIOFIT_PATCHES_CACHE, DOWNLOADED_PATCHES_DIR
)
DOWNLOADED_PATCHES_PATH = Path(
    os.getenv("BIOFIT_PATCHES_DOWNLOADED_PATCHES_PATH", DEFAULT_DOWNLOADED_PATCHES_PATH)
)

EXTRACTED_PATCHES_DIR = "extracted"
DEFAULT_EXTRACTED_PATCHES_PATH = os.path.join(
    DEFAULT_DOWNLOADED_PATCHES_PATH, EXTRACTED_PATCHES_DIR
)
EXTRACTED_PATCHES_PATH = Path(
    os.getenv("BIOFIT_PATCHES_EXTRACTED_PATCHES_PATH", DEFAULT_EXTRACTED_PATCHES_PATH)
)

PATCHES_FILENAME = "patches.json"
NO_PATCHES_FILENAME = "no_patches.json"
IS_CONDA = os.getenv("CONDA_PREFIX") is not None


PYARROW_AVAILABLE = False
PYARROW_VERSION = "N/A"

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
if PYARROW_AVAILABLE:
    try:
        PYARROW_VERSION = version.parse(importlib.metadata.version("pyarrow"))
        logger.info(f"pyarrow version {PYARROW_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

RPY2_AVAILABLE = False
RPY2_VERSION = "N/A"

RPY2_AVAILABLE = importlib.util.find_spec("rpy2") is not None
if RPY2_AVAILABLE:
    try:
        RPY2_VERSION = version.parse(importlib.metadata.version("rpy2"))
        logger.info(f"rpy2 version {RPY2_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

RPY2_ARROW_AVAILABLE = False
RPY2_ARROW_VERSION = "N/A"

RPY2_ARROW_AVAILABLE = importlib.util.find_spec("rpy2_arrow") is not None
if RPY2_ARROW_AVAILABLE:
    try:
        RPY2_ARROW_VERSION = version.parse(importlib.metadata.version("rpy2-arrow"))
        logger.info(f"rpy2 version {RPY2_ARROW_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

POLARS_VERSION = "N/A"
POLARS_AVAILABLE = False

POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None
if POLARS_AVAILABLE:
    try:
        POLARS_VERSION = version.parse(importlib.metadata.version("polars"))
        logger.info(f"Polars version {POLARS_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

DASK_VERSION = "N/A"
DASK_AVAILABLE = False

DASK_AVAILABLE = importlib.util.find_spec("dask") is not None
if DASK_AVAILABLE:
    try:
        DASK_VERSION = version.parse(importlib.metadata.version("dask"))
        logger.info(f"Dask version {DASK_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass


USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()


TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
    if TORCH_AVAILABLE:
        try:
            TORCH_VERSION = version.parse(importlib.metadata.version("torch"))
            logger.info(f"PyTorch version {TORCH_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling PyTorch because USE_TF is set")

TF_VERSION = "N/A"
TF_AVAILABLE = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
    if TF_AVAILABLE:
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for package in [
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        ]:
            try:
                TF_VERSION = version.parse(importlib.metadata.version(package))
            except importlib.metadata.PackageNotFoundError:
                continue
            else:
                break
        else:
            TF_AVAILABLE = False
    if TF_AVAILABLE:
        if TF_VERSION.major < 2:
            logger.info(
                f"TensorFlow found but with version {TF_VERSION}. `datasets` requires version 2 minimum."
            )
            TF_AVAILABLE = False
        else:
            logger.info(f"TensorFlow version {TF_VERSION} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")


JAX_VERSION = "N/A"
JAX_AVAILABLE = False

if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    JAX_AVAILABLE = (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("jaxlib") is not None
    )
    if JAX_AVAILABLE:
        try:
            JAX_VERSION = version.parse(importlib.metadata.version("jax"))
            logger.info(f"JAX version {JAX_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling JAX because USE_JAX is set to False")


USE_BEAM = os.environ.get("USE_BEAM", "AUTO").upper()
BEAM_VERSION = "N/A"
BEAM_AVAILABLE = False
if USE_BEAM in ENV_VARS_TRUE_AND_AUTO_VALUES:
    try:
        BEAM_VERSION = version.parse(importlib.metadata.version("apache_beam"))
        BEAM_AVAILABLE = True
        logger.info(f"Apache Beam version {BEAM_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass
else:
    logger.info("Disabling Apache Beam because USE_BEAM is set to False")


R_SCRIPTS = Path(__file__).parent / "integration/R/scripts"
BIOFIT_SKIP_R_DEPENDENCIES = (
    os.getenv("BIOFIT_SKIP_R_DEPENDENCIES", "true").upper() not in ENV_VARS_FALSE_VALUES
)
RECORDER_ENABLED = os.getenv("BIOFIT_RECORDER_ENABLED", "true").lower() == "true"

NO_IMBALANCE_ADJUSTMENT = os.getenv("BIOFIT_NO_IMBALANCE_ADJUSTMENT", "false") == "true"

PBAR_REFRESH_TIME_INTERVAL = 0.05
