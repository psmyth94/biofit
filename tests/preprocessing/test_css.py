import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import (
    is_biosets_available,
    is_polars_available,
    is_rpy2_available,
)
from biofit.preprocessing import CumulativeSumScaler
from biofit.preprocessing.scaling.css.plot_css import (
    CumulativeSumScalerPlotter,
    CumulativeSumScalerPlotterConfigForOTU,
)

from tests.utils import create_bioset

handler = DataHandler()

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "format",
    [
        "pandas",
        "polars",
        "numpy",
        "arrow",
        "dataset",
        "pandas_cached",
        "polars_cached",
        "numpy_cached",
        "arrow_cached",
        "dataset_cached",
    ],
)
def test_css(float_data, sample_metadata, format):
    format = format.replace("_cached", "")
    X, y = float_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    proc = CumulativeSumScaler()
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if is_rpy2_available():
        plotter = CumulativeSumScalerPlotter(
            config=CumulativeSumScalerPlotterConfigForOTU()
        )
    else:
        plotter = None
    input_columns = list(X.columns)
    if format == "numpy":
        data = handler.to_format(X, format)
        trans_data = proc.fit_transform(data, cache_dir=cache_dir)
        target = handler.to_format(y, format)
        if plotter is not None:
            plotter.plot(x1=data, x2=trans_data, y1=target, y2=target)

    else:
        if format == "dataset":
            if not is_biosets_available():
                pytest.skip("test requires biosets")
            from biosets.features import Abundance, BinClassLabel

            otu_dataset = create_bioset(
                X=X,
                y=y,
                sample_metadata=sample_metadata,
                with_feature_metadata=True,
                feature_type=Abundance,
                target_type=BinClassLabel,
            )

            # column does not need to be specified for dataset format
            trans_data = proc.fit_transform(otu_dataset, cache_dir=cache_dir)
            if plotter is not None:
                plotter.plot(otu_dataset, trans_data)
        else:
            if format == "polars" and not is_polars_available():
                pytest.skip("test requires polars")
            data = handler.to_format(otu_dataset, format)
            trans_data = proc.fit_transform(
                data,
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
            if plotter is not None:
                plotter.plot(
                    x1=data,
                    x2=trans_data,
                    input_columns1=input_columns,
                    input_columns2=input_columns,
                    label_name1="labels",
                    label_name2="labels",
                )
    # TODO: add assertions for the output
