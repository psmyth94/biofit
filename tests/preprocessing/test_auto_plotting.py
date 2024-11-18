import pandas as pd
import pytest
from biocore.data_handling import DataHandler

import biofit.config
from biofit.auto import AutoPlotter
from tests.utils import require_biosets, require_rpy2

handler = DataHandler()


pytestmark = pytest.mark.integration


@require_biosets
@require_rpy2
@pytest.mark.parametrize("format", ["dataset", "dataset_cached"])
def test_auto_plotting_otu(count_data, sample_metadata, format):
    from biosets.features import Abundance
    from datasets.features import Features

    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME

    otu_dataset = DataHandler.to_bioset(otu_dataset)
    otu_dataset._info.features = Features(
        {
            k: Abundance(dtype="int64") if k in X.columns else v
            for k, v in otu_dataset._info.features.items()
        }
    )
    proc = AutoPlotter.for_dataset("otu", path=cache_dir)
    proc.plot(otu_dataset, path=cache_dir)


@require_biosets
@require_rpy2
@pytest.mark.parametrize("format", ["dataset", "dataset_cached"])
def test_auto_plotting_snp(binary_data, sample_metadata, format):
    from biosets.features import Abundance
    from datasets.features import Features

    format = format.replace("_cached", "")
    X, y = binary_data
    snp_dataset = pd.concat([sample_metadata, X, y], axis=1)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME

    snp_dataset = DataHandler.to_bioset(snp_dataset)
    snp_dataset._info.features = Features(
        {
            k: Abundance(dtype="int64") if k in X.columns else v
            for k, v in snp_dataset._info.features.items()
        }
    )
    proc = AutoPlotter.for_dataset("snp", path=cache_dir)
    proc.plot(snp_dataset, path=cache_dir)
