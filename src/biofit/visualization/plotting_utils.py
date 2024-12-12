import sys
from os import PathLike
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from biocore import DataHandler
from biocore.utils import requires_backends
from sklearn.pipeline import Pipeline

from biofit.processing import SelectedColumnTypes
from biofit.utils.types import Unset


def is_in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        return False
    return False


def display_image_carousel(image_paths, format="png"):
    """
    Displays an image carousel with left and right arrow buttons in a Jupyter notebook.

    Parameters:
        image_paths (list): List of paths to images.
        output_dir (str): Directory to indicate in the log if limit is exceeded.
        max_images (int): Maximum number of images to include in the carousel.
    """
    requires_backends(display_image_carousel, "ipywidgets")
    import ipywidgets as widgets
    from IPython.display import Image, clear_output, display
    from ipywidgets import Button, HBox, Layout, VBox

    image_index = 0

    button_layout = Layout(width="100px")
    left_button = Button(description="Previous", layout=button_layout)
    right_button = Button(description="Next", layout=button_layout)
    image_box = widgets.Output()

    def show_image():
        """Helper function to show the image."""
        with image_box:
            clear_output(wait=True)
            display(
                Image(image_paths[image_index], embed=True, format=format, width=720)
            )

    def on_left_button_clicked(b):
        nonlocal image_index
        if image_index > 0:
            image_index -= 1
            show_image()

    def on_right_button_clicked(b):
        nonlocal image_index
        if image_index < len(image_paths) - 1:
            image_index += 1
            show_image()

    left_button.on_click(on_left_button_clicked)
    right_button.on_click(on_right_button_clicked)

    show_image()  # Show the first image initially

    navigation_box = HBox([left_button, right_button])
    display(VBox([navigation_box, image_box]))


def get_distinct_colors(n, colormap="nipy_spectral"):
    """
    Generate a list of n distinct colors using a specified colormap.
    Switched to 'nipy_spectral' for higher contrast and brightness differentiation.

    :param n: Number of distinct colors to generate.
    :param colormap: Name of the matplotlib colormap to use.
    :return: List of RGBA colors.
    """
    requires_backends(get_distinct_colors, "matplotlib")
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(colormap)
    return [cmap(i / n) for i in range(n)]


def generate_violin(
    x,
    y=None,
    column: SelectedColumnTypes = None,
    label_name: SelectedColumnTypes = None,
    xlab: str = Unset('"Labels"'),
    ylab: str = Unset('"Value"'),
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
    output_dir: Union[PathLike, str] = None,
):
    """Generates a violin plot of the input data.

    Args:
        x: Input data.
        other_x: Other input data.
        y: Target data.
        other_x: Other target data.
        **kwargs: Additional arguments.
    Returns:
        Plotter object.
    """
    from biofit.visualization.violin import ViolinPlotter

    return ViolinPlotter().plot(
        x,
        y,
        column=column,
        label_name=label_name,
        xlab=xlab,
        ylab=ylab,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def generate_scatterplot(
    x,
    y=None,
    group=None,
    xdata: SelectedColumnTypes = None,
    ydata: SelectedColumnTypes = None,
    groupby: SelectedColumnTypes = None,
    xlab: str = Unset("None"),
    ylab: str = Unset("None"),
    title: str = Unset('"Scatterplot"'),
    alpha: str = Unset("1"),
    col_set: str = Unset('"Set1"'),
    cols: List[str] = Unset("None"),
    xlog: str = Unset("None"),
    ylog: str = Unset("None"),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    from biofit.visualization.scatterplot import ScatterPlotter

    plotter = ScatterPlotter()
    return plotter.plot(
        x,
        y,
        group,
        xdata=xdata,
        ydata=ydata,
        groupby=groupby,
        xlab=xlab,
        ylab=ylab,
        title=title,
        alpha=alpha,
        col_set=col_set,
        cols=cols,
        xlog=xlog,
        ylog=ylog,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def generate_histogram(
    x,
    input_columns: SelectedColumnTypes = None,
    xlab: str = Unset('"X"'),
    ylab: str = Unset('"Frequency"'),
    title: str = Unset('"Histogram"'),
    bins: int = Unset("30"),
    font_size: int = Unset("8"),
    col_fill: str = Unset('"grey40"'),
    col_outline: str = Unset('"white"'),
    xlog: Optional[str] = Unset("None"),
    ylog: Optional[str] = Unset("None"),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    """Generates a histogram of the input data.

    Args:
        x: Input data.
        other_x: Other input data.
        y: Target data.
        other_x: Other target data.
        **kwargs: Additional arguments.
    Returns:
        Plotter object.
    """
    from biofit.visualization.histogram import HistogramPlotter

    plotter = HistogramPlotter()

    return plotter.plot(
        x,
        input_columns=input_columns,
        xlab=xlab,
        ylab=ylab,
        title=title,
        bins=bins,
        font_size=font_size,
        col_fill=col_fill,
        col_outline=col_outline,
        xlog=xlog,
        ylog=ylog,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def generate_barplot(
    x,
    y=None,
    group=None,
    label_name: SelectedColumnTypes = None,
    value_name=None,
    groupby: Optional[str] = None,
    xlab: Optional[str] = Unset("None"),
    ylab: Optional[str] = Unset("None"),
    title: str = Unset('"Bar Plot"'),
    col_set: str = Unset('"Set1"'),
    col_labels: str = Unset('"black"'),
    col_outline: str = Unset('"grey30"'),
    cols: Optional[List[str]] = Unset("None"),
    prop: bool = Unset("False"),
    add_count_lab: bool = Unset("True"),
    vars_as_entered: bool = Unset("False"),
    legend_position: str = Unset('"top"'),
    font_size: float = Unset("3.25"),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    """Generates a bar plot of the input data.

    Args:
        x: Input data.
        other_x: Other input data.
        y: Target data.
        other_x: Other target data.
        **kwargs: Additional arguments.
    Returns:
        Plotter object.
    """
    from biofit.visualization.barplot import BarPlotter

    plotter = BarPlotter()

    return plotter.plot(
        x,
        y,
        group,
        label_name=label_name,
        value_name=value_name,
        groupby=groupby,
        xlab=xlab,
        ylab=ylab,
        title=title,
        col_set=col_set,
        col_labels=col_labels,
        col_outline=col_outline,
        cols=cols,
        prop=prop,
        add_count_lab=add_count_lab,
        vars_as_entered=vars_as_entered,
        legend_position=legend_position,
        font_size=font_size,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def generate_comparison_histogram(
    x1,
    x2=None,
    column1: str = None,
    column2: str = None,
    xlab: Optional[str] = Unset("None"),
    ylab: str = Unset("None"),
    title: str = Unset("None"),
    bins: int = Unset("None"),
    alpha: float = Unset("None"),
    legend_title: str = Unset("None"),
    legend_position: str = Unset("None"),
    subplot_title1: str = Unset("None"),
    subplot_title2: str = Unset("None"),
    col_set: str = Unset("None"),
    cols: Optional[List[str]] = Unset("None"),
    xlog: Optional[bool] = Unset("None"),
    ylog: Optional[bool] = Unset("None"),
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
    output_dir=None,
):
    """Generates a comparison histogram of the input data.

    Args:
        x: Input data.
        other_x: Other input data.
        y: Target data.
        other_x: Other target data.
        **kwargs: Additional arguments.
    Returns:
        Plotter object.
    """
    from biofit.visualization.histogram import ComparisonHistogramPlotter

    plotter = ComparisonHistogramPlotter()

    return plotter.plot(
        x1,
        x2=x2,
        column1=column1,
        column2=column2,
        xlab=xlab,
        ylab=ylab,
        title=title,
        bins=bins,
        alpha=alpha,
        legend_title=legend_title,
        legend_position=legend_position,
        col_set=col_set,
        cols=cols,
        subplot_title1=subplot_title1,
        subplot_title2=subplot_title2,
        xlog=xlog,
        ylog=ylog,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def plot_correlation(
    X,
    y=None,
    group=None,
    input_columns=None,
    target_column=None,
    groupby: Optional[str] = None,
    precomputed=False,
    method="auto",
    label_name: str = Unset('"None"'),
    value_name: Optional[str] = Unset("None"),
    top_k: int = Unset("30"),
    xlab: Optional[str] = Unset("None"),
    ylab: Optional[str] = Unset("None"),
    title: str = Unset('"Bar Plot"'),
    col_set: str = Unset('"Set1"'),
    cols: Optional[List[str]] = Unset("None"),
    prop: bool = Unset("False"),
    add_count_lab: bool = Unset("True"),
    vars_as_entered: bool = Unset("False"),
    legend_position: str = Unset('"top"'),
    font_size: float = Unset("3.25"),
    output_dir=None,
    file_name=None,
):
    """Plot the correlation matrix of the input data.

    Args:
        X: Input data
        y: Target variable
        input_columns: Columns to filter
        target_column: Target column
        top_k: Number of top features to plot
        label_name: Name of the labels
        value_name: Name of the values
        groupby: Groupby column
        xlab: X-axis label
        ylab: Y-axis label
        title: Plot title
        col_set: Color set
        cols: Columns to plot
        prop: Whether to plot proportions
        add_count_lab: Whether to add count labels
        vars_as_entered: Whether the variables are entered as is
        legend_position: Position of the legend
        font_size: Font size
    Returns:
        SampleFiltered data.
    """
    if not precomputed:
        from biofit.stat import CorrelationStat

        corr_stat = CorrelationStat(method=method)
        corrs = corr_stat.fit_transform(
            X, y, input_columns, target_column, output_format="pandas"
        )
        name_map = {
            "pearsonr": "Pearson Correlation",
            "spearmanr": "Spearman Correlation",
            "kendalltau": "Kendall Correlation",
            "pointbiserialr": "Point Biserial Correlation",
        }
        value_name = name_map.get(corr_stat.config.method, "Correlation")
        if groupby is None or isinstance(groupby, Unset):
            groupby = "Features"
        corrs = corrs.melt(var_name=groupby, value_name=value_name)
        # drop na
        corrs = corrs.dropna()
        sorted_inds = np.argsort(np.abs(corrs[value_name]))[::-1]
        if groupby is None or isinstance(top_k, Unset):
            top_k = 30
        corrs = corrs.iloc[sorted_inds[:top_k]]
        return generate_barplot(
            corrs,
            None,
            None,
            label_name=label_name,
            value_name=value_name,
            groupby=groupby,
            xlab=xlab,
            ylab=ylab,
            title=title,
            col_set=col_set,
            cols=cols,
            prop=prop,
            add_count_lab=add_count_lab,
            vars_as_entered=vars_as_entered,
            legend_position=legend_position,
            font_size=font_size,
            output_dir=output_dir,
        )

    return generate_barplot(
        X,
        y,
        None,
        label_name=label_name,
        value_name=value_name,
        groupby=groupby,
        xlab=xlab,
        ylab=ylab,
        title=title,
        col_set=col_set,
        cols=cols,
        prop=prop,
        add_count_lab=add_count_lab,
        vars_as_entered=vars_as_entered,
        legend_position=legend_position,
        font_size=font_size,
        output_dir=output_dir,
    )


def plot_feature_distribution(
    X,
    input_columns: SelectedColumnTypes = None,
    value_name=None,
    aggregate=None,
    aggregate_kwargs={},
    xlab: str = Unset('"X"'),
    ylab: str = Unset('"Frequency"'),
    title: str = Unset('"Histogram"'),
    bins: int = Unset("30"),
    font_size: int = Unset("8"),
    col_fill: str = Unset('"grey40"'),
    col_outline: str = Unset('"white"'),
    xlog: Optional[str] = Unset("None"),
    ylog: Optional[str] = Unset("None"),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    """Plot the feature distribution of the input data.

    Args:
        X: Input data
        columns: Columns to plot
        aggregate: Aggregate each feature by: 'mean', 'median', 'sum', 'std', 'var', 'min', 'max', or 'presence'
        **kwargs: Additional keyword arguments

    Returns:
        Plot object.
    """
    x_dims = DataHandler.get_shape(X)
    precomputed = aggregate is None and (
        value_name is not None or len(x_dims) == 1 or x_dims[1] == 1
    )
    value_name = value_name or "Presence"

    if aggregate == "presence":
        from biofit.stat import ColumnMissingnessStat

        num_rows = DataHandler.get_shape(X)[0]
        missingness = ColumnMissingnessStat(
            input_columns=input_columns, **aggregate_kwargs
        ).fit_transform(X, output_format="pandas")
        data = num_rows - missingness
    elif aggregate == "sum":
        from biofit.stat import ColumnSumStat

        data = ColumnSumStat(
            input_columns=input_columns, **aggregate_kwargs
        ).fit_transform(X, output_format="pandas")
    elif aggregate == "mean":
        from biofit.stat import ColumnMeanStat

        data = ColumnMeanStat(
            input_columns=input_columns, **aggregate_kwargs
        ).fit_transform(X, output_format="pandas")

    if (
        input_columns is None
        and "biosets" in sys.modules
        and isinstance(X, getattr(sys.modules["biosets"], "Bioset"))
    ):
        from biosets import get_data

        data = get_data(X)
    else:
        data = X

    data = DataHandler.to_pandas(data)

    if input_columns:
        data = data[input_columns]

    if not precomputed:
        data = data.melt(value_name=value_name)

    return generate_histogram(
        data,
        input_columns=input_columns,
        xlab=xlab,
        ylab=ylab,
        title=title,
        bins=bins,
        font_size=font_size,
        col_fill=col_fill,
        col_outline=col_outline,
        xlog=xlog,
        ylog=ylog,
        output_dir=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def compare_feature_distributions(
    X1,
    X2,
    columns1: SelectedColumnTypes = None,
    columns2: SelectedColumnTypes = None,
    value_name=None,
    aggregate=None,
    aggregate_kwargs={},
    xlab: Optional[str] = Unset("None"),
    ylab: str = Unset("None"),
    title: str = Unset("None"),
    bins: int = Unset("None"),
    alpha: float = Unset("None"),
    legend_title: str = Unset("None"),
    legend_position: str = Unset("None"),
    subplot_title1: str = Unset("None"),
    subplot_title2: str = Unset("None"),
    col_set: str = Unset("None"),
    cols: Optional[List[str]] = Unset("None"),
    xlog: Optional[bool] = Unset("None"),
    ylog: Optional[bool] = Unset("None"),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    """Compare the feature distribution of the input data.

    Args:
        X1: Input data 1
        X2: Input data 2
        columns1: Columns to plot for data 1
        columns2: Columns to plot for data 2. If not provided, columns1 will be used for data 2.
        aggregate: Aggregate each feature by: 'mean', 'median', 'sum', 'std', 'var', 'min', 'max', or 'presence'
        **kwargs: Additional keyword arguments
    Returns:
        Plot object.
    """
    x1_dims = DataHandler.get_shape(X1)
    x2_dims = DataHandler.get_shape(X2)
    precomputed = (
        aggregate is None
        and (value_name is not None or len(x1_dims) == 1 or x1_dims[1] == 1)
        and (value_name is not None or len(x2_dims) == 1 or x2_dims[1] == 1)
    )

    if columns1 and not columns2:
        columns2 = columns1

    if aggregate == "presence":
        from biofit.stat import ColumnMissingnessStat

        value_name = value_name or "Presence"
        num_rows1 = DataHandler.get_shape(X1)[0]
        num_rows2 = DataHandler.get_shape(X2)[0]

        missingness1 = ColumnMissingnessStat(
            input_columns=columns1, **aggregate_kwargs
        ).fit_transform(X1, output_format="pandas")
        missingness2 = ColumnMissingnessStat(
            input_columns=columns2, **aggregate_kwargs
        ).fit_transform(X2, output_format="pandas")

        data1 = num_rows1 - missingness1
        data2 = num_rows2 - missingness2
    elif aggregate == "sum":
        from biofit.stat import ColumnSumStat

        value_name = value_name or "Sum"
        data1 = ColumnSumStat(input_columns=columns1, **aggregate_kwargs).fit_transform(
            X1, output_format="pandas"
        )
        data2 = ColumnSumStat(input_columns=columns2, **aggregate_kwargs).fit_transform(
            X2, output_format="pandas"
        )
    elif aggregate == "mean":
        from biofit.stat import ColumnMeanStat

        value_name = value_name or "Mean"
        data1 = ColumnMeanStat(
            input_columns=columns1, **aggregate_kwargs
        ).fit_transform(X1, output_format="pandas")
        data2 = ColumnMeanStat(
            input_columns=columns2, **aggregate_kwargs
        ).fit_transform(X2, output_format="pandas")
    else:
        data1 = X1
        data2 = X2

    value_name = value_name or "Value"

    data1 = DataHandler.to_pandas(data1)
    data2 = DataHandler.to_pandas(data2)

    if columns1:
        data1 = data1[columns1]

    if columns2:
        data2 = data2[columns2]

    if not precomputed:
        if data1.shape[1] > 2:
            data1 = data1.melt(value_name=value_name)

        if data2.shape[1] > 2:
            data2 = data2.melt(value_name=value_name)

    return generate_comparison_histogram(
        data1,
        data2,
        column1=value_name,
        column2=value_name,
        xlab=xlab,
        ylab=ylab,
        title=title,
        bins=bins,
        alpha=alpha,
        legend_title=legend_title,
        legend_position=legend_position,
        col_set=col_set,
        cols=cols,
        subplot_title1=subplot_title1,
        subplot_title2=subplot_title2,
        xlog=xlog,
        ylog=ylog,
        output_dir=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def plot_sample_distribution(
    X,
    input_columns: SelectedColumnTypes = None,
    value_name=None,
    aggregate=None,
    aggregate_kwargs={},
    xlab: str = Unset('"X"'),
    ylab: str = Unset('"Frequency"'),
    title: str = Unset('"Histogram"'),
    bins: int = Unset("30"),
    font_size: int = Unset("8"),
    col_fill: str = Unset('"grey40"'),
    col_outline: str = Unset('"white"'),
    xlog: Optional[str] = Unset("None"),
    ylog: Optional[str] = Unset("None"),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    """Plot the feature distribution of the input data.

    Args:
        X: Input data
        columns: Columns to plot
        aggregate: Aggregate each feature by: 'mean', 'median', 'sum', 'std', 'var', 'min', 'max', or 'presence'
        **kwargs: Additional keyword arguments

    Returns:
        Plot object.
    """
    x_dims = DataHandler.get_shape(X)
    precomputed = aggregate is None and (
        value_name is not None or len(x_dims) == 1 or x_dims[1] == 1
    )

    if aggregate == "presence":
        from biofit.stat import RowMissingnessStat

        value_name = value_name or "Presence"
        num_rows = DataHandler.get_shape(X)[0]
        missingness = RowMissingnessStat(
            input_columns=input_columns, **aggregate_kwargs
        ).fit_transform(X, output_format="pandas")
        data = num_rows - missingness
    elif aggregate == "sum":
        from biofit.stat import RowSumStat

        value_name = value_name or "Sum"
        data = RowSumStat(
            input_columns=input_columns, **aggregate_kwargs
        ).fit_transform(X, output_format="pandas")
    elif aggregate == "mean":
        from biofit.stat import RowMeanStat

        value_name = value_name or "Mean"
        data = RowMeanStat(
            input_columns=input_columns, **aggregate_kwargs
        ).fit_transform(X, output_format="pandas")

    value_name = value_name or "Value"

    if (
        input_columns is None
        and "biosets" in sys.modules
        and isinstance(X, getattr(sys.modules["biosets"], "Bioset"))
    ):
        from biosets import get_data

        data = get_data(X)
    else:
        data = X

    data = DataHandler.to_pandas(data)

    if input_columns:
        data = data[input_columns]

    if not precomputed:
        if data.shape[1] > 2:
            data = data.melt(value_name=value_name)

    return generate_histogram(
        data,
        input_columns=value_name,
        xlab=xlab,
        ylab=ylab,
        title=title,
        bins=bins,
        font_size=font_size,
        col_fill=col_fill,
        col_outline=col_outline,
        xlog=xlog,
        ylog=ylog,
        output_dir=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def compare_sample_distributions(
    X1,
    X2,
    columns1: SelectedColumnTypes = None,
    columns2: SelectedColumnTypes = None,
    value_name=None,
    aggregate=None,
    aggregate_kwargs={},
    xlab: Optional[str] = Unset("None"),
    ylab: str = Unset("None"),
    title: str = Unset("None"),
    bins: int = Unset("None"),
    alpha: float = Unset("None"),
    legend_title: str = Unset("None"),
    legend_position: str = Unset("None"),
    subplot_title1: str = Unset("None"),
    subplot_title2: str = Unset("None"),
    col_set: str = Unset("None"),
    cols: Optional[List[str]] = Unset("None"),
    xlog: Optional[bool] = Unset("None"),
    ylog: Optional[bool] = Unset("None"),
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
    output_dir=None,
):
    """Compare the feature distribution of the input data.

    Args:
        X1: Input data 1
        X2: Input data 2
        columns1: Columns to plot for data 1
        columns2: Columns to plot for data 2. If not provided, columns1 will be used for data 2.
        aggregate: Aggregate each feature by: 'mean', 'median', 'sum', 'std', 'var', 'min', 'max', or 'presence'
        **kwargs: Additional keyword arguments

    Returns:
        Plot object.
    """
    x1_dims = DataHandler.get_shape(X1)
    x2_dims = DataHandler.get_shape(X2)
    precomputed = (
        aggregate is None
        and (value_name is not None or len(x1_dims) == 1 or x1_dims[1] == 1)
        and (value_name is not None or len(x2_dims) == 1 or x2_dims[1] == 1)
    )

    if columns1 and not columns2:
        columns2 = columns1

    if aggregate == "presence":
        from biofit.stat import RowMissingnessStat

        value_name = value_name or "Presence"
        num_cols1 = x1_dims[1] if len(x1_dims) == 2 else 1
        num_cols2 = x2_dims[1] if len(x2_dims) == 2 else 1

        missingness1 = RowMissingnessStat(
            input_columns=columns1, keep_unused_columns=False, **aggregate_kwargs
        ).fit_transform(X1, output_format="pandas")
        missingness2 = RowMissingnessStat(
            input_columns=columns2, keep_unused_columns=False, **aggregate_kwargs
        ).fit_transform(X2, output_format="pandas")

        data1 = num_cols1 - missingness1
        data2 = num_cols2 - missingness2
        data1.columns = [value_name]
        data2.columns = [value_name]
    elif aggregate == "sum":
        from biofit.stat import RowSumStat

        value_name = value_name or "Sum"
        data1 = RowSumStat(
            input_columns=columns1, keep_unused_columns=False, **aggregate_kwargs
        ).fit_transform(X1, output_format="pandas")
        data2 = RowSumStat(
            input_columns=columns2, keep_unused_columns=False, **aggregate_kwargs
        ).fit_transform(X2, output_format="pandas")
        data1.columns = [value_name]
        data2.columns = [value_name]
    elif aggregate == "mean":
        from biofit.stat import RowMeanStat

        value_name = value_name or "Mean"
        data1 = RowMeanStat(
            input_columns=columns1, keep_unused_columns=False, **aggregate_kwargs
        ).fit_transform(X1, output_format="pandas")
        data2 = RowMeanStat(
            input_columns=columns2, keep_unused_columns=False, **aggregate_kwargs
        ).fit_transform(X2, output_format="pandas")
        data1.columns = [value_name]
        data2.columns = [value_name]

    value_name = value_name or "Value"

    data1 = DataHandler.to_pandas(data1)
    data2 = DataHandler.to_pandas(data2)

    if columns1:
        data1 = data1[columns1]

    if columns2:
        data2 = data2[columns2]

    if not precomputed:
        if data1.shape[1] > 1:
            data1 = data1.melt(value_name=value_name)
        else:
            data1.columns = [value_name]

        if data2.shape[1] > 1:
            data2 = data2.melt(value_name=value_name)
        else:
            data1.columns = [value_name]

    if value_name not in data1.columns:
        raise ValueError(
            f"Column '{value_name}' not found in data1. Found columns: {data1.columns}"
        )
    if value_name not in data2.columns:
        raise ValueError(
            f"Column '{value_name}' not found in data2. Found columns: {data2.columns}"
        )

    return generate_comparison_histogram(
        data1,
        data2,
        column1=value_name,
        column2=value_name,
        xlab=xlab,
        ylab=ylab,
        title=title,
        bins=bins,
        alpha=alpha,
        legend_title=legend_title,
        legend_position=legend_position,
        col_set=col_set,
        cols=cols,
        subplot_title1=subplot_title1,
        subplot_title2=subplot_title2,
        xlog=xlog,
        ylog=ylog,
        output_dir=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def plot_dimension_reduction(
    X,
    labels=None,
    group=None,
    input_columns: SelectedColumnTypes = None,
    label_column: SelectedColumnTypes = None,
    group_column: SelectedColumnTypes = None,
    method=None,
    method_kwargs={},
    title: str = Unset('"Dimension Reduction Plot"'),
    colormap: str = Unset('"nipy_spectral"'),
    n_components: int = Unset("3"),
    dimension_reducer=None,
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
    show=True,
):
    """Plot the dimension reduction plot.

    Args:
        X: Input data
        labels: Labels for the data
        group: Group for the data
        dimension_reducer: Dimension reducer object
        **kwargs: Additional keyword arguments

    Returns:
        Plot object.
    """
    requires_backends(plot_sample_metadata, ["matplotlib", "seaborn"])
    if labels is None and label_column is None:
        if "biosets" in sys.modules and isinstance(
            X, getattr(sys.modules["biosets"], "Bioset")
        ):
            from biosets import get_target

            labels = get_target(X)
    elif label_column:
        labels = DataHandler.select_columns(X, label_column)
        if "biosets" in sys.modules and isinstance(
            X, getattr(sys.modules["biosets"], "Bioset")
        ):
            from biosets import decode

            labels = decode(labels)

    if input_columns is not None:
        X = DataHandler.select_columns(X, input_columns)
        input_columns = None
    if method is None:
        from biofit.preprocessing import PCAFeatureExtractor

        method = PCAFeatureExtractor
    if isinstance(method, type):
        method = method()
    if isinstance(method, str):
        from biofit.auto.processing_auto import AutoPreprocessor

        if method_kwargs is None or not isinstance(method_kwargs, dict):
            method_kwargs = {}

        method = AutoPreprocessor.from_processor(method, **method_kwargs)
    if not method.is_fitted:
        data = method.fit_transform(X, load_from_cache_file=False)
    else:
        data = method.transform(X)

    from biofit.visualization.dimension_reduction import DimensionReductionPlotter

    return DimensionReductionPlotter().plot(
        data,
        labels,
        group,
        input_columns=input_columns,
        label_column=label_column,
        group_column=group_column,
        n_components=n_components,
        dimension_reducer=method,
        show=show,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def get_feature_importances(models, label_names=None):
    if not isinstance(models, list):
        models = [models]
    tables = []
    for fold, model in enumerate(models):
        if isinstance(model, Pipeline):
            _m = model[-1]
        else:
            _m = model
        features = _m.config.feature_names_in_
        feature_importances = _m.feature_importances_
        classes = label_names
        if classes is None:
            if hasattr(_m, "config") and hasattr(_m.config, "class_names"):
                classes = _m.config.class_names
            if classes is None:
                classes = getattr(_m, "classes_", None)

        if classes is not None and (
            (len(feature_importances) // len(classes)) == len(features)
        ):
            # this means its a one vs all classifier
            _tables = []
            for i, c in enumerate(classes):
                _tables.append(
                    pd.Series(
                        feature_importances[
                            i * len(features) : (i + 1) * len(features)
                        ],
                        index=features,
                        name=f"importances_{fold + 1}_{c}_vs_all",
                    )
                )
            tables.extend(_tables)
        else:
            tables.append(
                pd.DataFrame(
                    {
                        f"importances_{fold + 1}": feature_importances,
                    },
                    index=features,
                )
            )

    if len(tables) == 1:
        return tables[0]

    # Concatenate tables horizontally
    return pd.concat(tables, axis=1, ignore_index=False, copy=False)


def plot_feature_importance(
    feature_importances,
    models=None,
    X=None,
    y=None,
    sample_metadata=None,
    feature_metadata: dict = None,
    input_columns: SelectedColumnTypes = None,
    target_columns: SelectedColumnTypes = None,
    sample_metadata_columns: SelectedColumnTypes = None,
    sample_column: str = None,
    label_names: List[str] = None,
    plot_top: int = Unset("15"),
    feature_meta_name: str = Unset("None"),
    feature_column: str = Unset("None"),
    cols: List[str] = Unset("None"),
    colHeat: List[str] = Unset("None"),
    dat_log: str = Unset("field(default=None, init=True, repr=False)"),
    show_column_names: bool = Unset("False"),
    scale_legend_title: str = Unset('"Value"'),
    column_title: str = Unset('"Samples"'),
    row_title: str = Unset('"Features"'),
    plot_title: str = Unset('"Values"'),
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
    show=True,
):
    """Plot the feature importance of the input data.

    Args:
        X: Input data
        models: List of models
        y: Target variable
        sample_metadata: Sample metadata
        feature_metadata: Feature metadata
    Returns:
        Plot object.
    """

    from biofit.visualization.feature_importance import FeatureImportancePlotter

    if feature_importances is None and models is not None:
        feature_importances = get_feature_importances(models, label_names)
        feature_importances = feature_importances.reset_index(names=["features"])
    elif feature_importances is None:
        raise ValueError("feature_importances or models must be provided")

    FeatureImportancePlotter().plot(
        X,
        y=y,
        sample_metadata=sample_metadata,
        input_columns=input_columns,
        target_columns=target_columns,
        sample_metadata_columns=sample_metadata_columns,
        feature_importances=feature_importances,
        feature_metadata=feature_metadata,
        plot_top=plot_top,
        feature_meta_name=feature_meta_name,
        sample_column=sample_column,
        feature_column=feature_column,
        cols=cols,
        colHeat=colHeat,
        dat_log=dat_log,
        show_column_names=show_column_names,
        scale_legend_title=scale_legend_title,
        column_title=column_title,
        row_title=row_title,
        plot_title=plot_title,
        show=show,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )


def plot_sample_metadata(
    sample_metadata,
    outcome_data=None,
    sample_metadata_columns: Optional[SelectedColumnTypes] = None,
    outcome_column: Optional[SelectedColumnTypes] = None,
    output_dir: str = None,
    device: str = "pdf",
    fingerprint: str = None,
    unused_columns: SelectedColumnTypes = None,
    raise_if_missing: bool = True,
):
    requires_backends(plot_sample_metadata, ["rpy2"])
    from biofit.visualization.sample_metadata import SampleMetadataPlotter

    SampleMetadataPlotter().plot(
        sample_metadata,
        outcome_data,
        sample_metadata_columns=sample_metadata_columns,
        outcome_column=outcome_column,
        path=output_dir,
        device=device,
        fingerprint=fingerprint,
        unused_columns=unused_columns,
        raise_if_missing=raise_if_missing,
    )
