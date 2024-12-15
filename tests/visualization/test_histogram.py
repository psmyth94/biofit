import os

import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available
from biofit.visualization.histogram import HistogramPlotter

from tests.utils import create_bioset, require_matplotlib

pytestmark = pytest.mark.unit


@require_matplotlib
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
def test_histogram(count_data, sample_metadata, format):
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    output_dir = biofit.config.BIOFIT_CACHE_HOME
    plotter = HistogramPlotter(
        xlab="X-test",
        ylab="Y-test",
        title="Histogram-test",
        bins=10,
        font_size=8,
        col_fill="light grey",
        col_outline="white",
        xlog=None,
        ylog="log2",
    )
    input_columns = list(X.columns)
    if format == "numpy":
        plot = plotter.plot(
            DataHandler.to_format(X, format),
            output_dir=output_dir,
        )
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
            plot = plotter.plot(
                otu_dataset,
                xlab="X-test",
                ylab="Y-test",
                title="Histogram-test",
                output_dir=output_dir,
            )

        else:
            if format == "polars" and not is_polars_available():
                pytest.skip("test requires polars")
            data = DataHandler.to_format(otu_dataset, format)
            plot = plotter.plot(
                x=data,
                input_columns=input_columns,
                output_dir=output_dir,
            )

    import matplotlib.pyplot as plt

    assert isinstance(plot, plt.Axes)

    # make sure the titles are right
    assert plot.get_title() == "Histogram-test"
    assert plot.get_xlabel() == "X-test"
    assert plot.get_ylabel() == "Y-test"
    assert plot.get_xscale() == "linear"
    assert plot.get_yscale() == "log"

    # make sure the bins are right
    assert len(plot.patches) == 10
    assert plot.patches[0].get_facecolor() == (
        0.8470588235294118,
        0.8627450980392157,
        0.8392156862745098,
        0.6,
    )
    assert plot.patches[0].get_edgecolor() == (1.0, 1.0, 1.0, 0.6)

    # make sure the directory isn't empty and the plot is saved
    files = os.listdir(output_dir)
    assert len(files) == 1
    assert files[0].endswith(".png")
    assert os.path.getsize(os.path.join(output_dir, files[0])) > 0

    # delete the file
    os.remove(os.path.join(output_dir, files[0]))

    plt.close()
