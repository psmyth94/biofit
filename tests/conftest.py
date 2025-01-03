from pathlib import Path

import pytest
from biocore.utils.import_util import is_biosets_available, is_datasets_available
from biofit.utils.logging import silence

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './mock_packages')))

pytest_plugins = ["tests.fixtures.files", "tests.fixtures.fsspec"]


def pytest_collection_modifyitems(config, items):
    # Mark tests as "unit" by default if not marked as "integration" (or already marked as "unit")
    for item in items:
        if any(marker in item.keywords for marker in ["integration", "unit"]):
            continue
        item.add_marker(pytest.mark.unit)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "torchaudio_latest: mark test to run with torchaudio>=0.12"
    )


@pytest.fixture(autouse=True)
def set_test_cache_config(tmp_path_factory, monkeypatch):
    test_cache_home = tmp_path_factory.getbasetemp() / "cache"
    test_patches_cache = test_cache_home / "patches"
    test_datasets_cache = test_cache_home / "datasets"
    test_processors_cache = test_cache_home / "processors"
    test_modules_cache = test_cache_home / "modules"
    monkeypatch.setattr("biofit.config.BIOFIT_CACHE_HOME", Path(test_cache_home))
    monkeypatch.setattr("biofit.config.BIOFIT_PATCHES_CACHE", Path(test_patches_cache))
    test_downloaded_datasets_path = test_datasets_cache / "downloads"
    test_extracted_datasets_path = test_datasets_cache / "downloads" / "extracted"
    if is_biosets_available():
        monkeypatch.setattr(
            "biosets.config.BIOSETS_DATASETS_CACHE", Path(test_datasets_cache)
        )

        monkeypatch.setattr(
            "biosets.config.DOWNLOADED_BIOSETS_PATH",
            str(test_downloaded_datasets_path),
        )

        monkeypatch.setattr(
            "biosets.config.EXTRACTED_BIOSETS_PATH",
            str(test_extracted_datasets_path),
        )

    if is_datasets_available():
        monkeypatch.setattr(
            "datasets.config.HF_DATASETS_CACHE", str(test_datasets_cache)
        )
        monkeypatch.setattr("datasets.config.HF_MODULES_CACHE", str(test_modules_cache))
        monkeypatch.setattr(
            "datasets.config.DOWNLOADED_DATASETS_PATH",
            str(test_downloaded_datasets_path),
        )
        monkeypatch.setattr(
            "datasets.config.EXTRACTED_DATASETS_PATH", str(test_extracted_datasets_path)
        )

    monkeypatch.setattr(
        "biofit.config.BIOFIT_PROCESSORS_CACHE", Path(test_processors_cache)
    )
    monkeypatch.setattr("biofit.config.BIOFIT_MODULES_CACHE", Path(test_modules_cache))


# @pytest.fixture(autouse=True, scope="session")
# def disable_tqdm_output():
#     disable_progress_bar()


# @pytest.fixture(autouse=True, scope="session")
# def set_info_verbosity():
#     set_verbosity_info()


@pytest.fixture(autouse=True, scope="session")
def silence_ouput():
    silence()


@pytest.fixture(autouse=True)
def set_update_download_counts_to_false(monkeypatch):
    # don't take tests into account when counting downloads
    if is_datasets_available():
        monkeypatch.setattr("datasets.config.HF_UPDATE_DOWNLOAD_COUNTS", False)


@pytest.fixture
def set_sqlalchemy_silence_uber_warning(monkeypatch):
    # Required to suppress RemovedIn20Warning when feature(s) are not compatible with SQLAlchemy 2.0
    # To be removed once SQLAlchemy 2.0 supported
    try:
        monkeypatch.setattr("sqlalchemy.util.deprecations.SILENCE_UBER_WARNING", True)
    except AttributeError:
        pass


@pytest.fixture(autouse=True, scope="session")
def zero_time_out_for_remote_code():
    if is_datasets_available():
        import datasets.config

        datasets.config.TIME_OUT_REMOTE_CODE = 0
