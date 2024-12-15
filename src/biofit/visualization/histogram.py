import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Type

import numpy as np
import pandas as pd
from biocore.utils.import_util import requires_backends

import biofit.config as config
from biofit.integration.biosets import get_feature
from biofit.integration.R import RCaller
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils.types import Unset
from biofit.visualization.plotting import BasePlotter, PlotterConfig
from biofit.visualization.plotting_utils import (
    color_name_to_rgb,
    log_transformation,
    prepare_axis_label,
)


@dataclass
class HistogramConfig(PlotterConfig):
    processor_type: str = field(default="scaling", init=False, repr=False)
    _unused_feature_types: List[Type] = field(
        default=get_feature("METADATA_FEATURE_TYPES"),
        init=False,
        repr=False,
    )

    xlab: str = "X"
    ylab: str = "Frequency"
    title: str = "Histogram"
    bins: int = 30
    font_size: int = 8
    col_fill: str = "light grey"
    col_outline: str = "white"
    x1_name: str = "Before"
    x2_name: str = "After"
    xlog: Optional[str] = None
    ylog: Optional[str] = None

    def __post_init__(self):
        if self.xlog not in [
            None,
            "log2",
            "log10",
            "log",
            "log2_1p",
            "log10_1p",
            "log1p",
        ]:
            raise ValueError(
                f"Invalid value for xlog: {self.xlog}. Must be one of: None, 'log2', 'log10', 'log', 'log2_1p', 'log10_1p', 'log1p'"
            )
        if self.ylog not in [
            None,
            "log2",
            "log10",
            "log",
            "log2_1p",
            "log10_1p",
            "log1p",
        ]:
            raise ValueError(
                f"Invalid value for ylog: {self.ylog}. Must be one of: None, 'log2', 'log10', 'log', 'log2_1p', 'log10_1p', 'log1p'"
            )


class HistogramPlotter(BasePlotter):
    _config_class = HistogramConfig
    config: HistogramConfig

    def __init__(
        self,
        xlab: str = Unset('"X"'),
        ylab: str = Unset('"Frequency"'),
        title: str = Unset('"Histogram"'),
        bins: int = Unset("30"),
        font_size: int = Unset("8"),
        col_fill: str = Unset('"grey40"'),
        col_outline: str = Unset('"white"'),
        xlog: Optional[str] = Unset("None"),
        ylog: Optional[str] = Unset("None"),
        install_missing: bool = None,
        config: Optional[HistogramConfig] = None,
    ):
        super().__init__(
            xlab=xlab,
            ylab=ylab,
            title=title,
            bins=bins,
            font_size=font_size,
            col_fill=col_fill,
            col_outline=col_outline,
            xlog=xlog,
            ylog=ylog,
            config=config,
            install_missing=install_missing,
        )

    def plot(
        self,
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
        path: str = None,
        device: str = "png",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
        output_dir=None,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(input_columns)
        return self._plot(
            x,
            xlab=xlab,
            ylab=ylab,
            title=title,
            bins=bins,
            font_size=font_size,
            col_fill=col_fill,
            col_outline=col_outline,
            xlog=xlog,
            ylog=ylog,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
            output_dir=output_dir,
        )

    def plot_pandas(self, X):
        requires_backends("_plot_pandas", "matplotlib")
        import matplotlib.pyplot as plt

        # colors options

        if isinstance(X, pd.DataFrame):
            if len(X.columns) > 1:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError(
                        "No numeric columns found in the provided DataFrame."
                    )
                X = X[numeric_cols[0]]
            else:
                X = X.iloc[:, 0]

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Data must be numeric to plot a histogram.")

        if self.config.xlog is not None:
            self.config.xlab = prepare_axis_label(self.config.xlab, self.config.xlog)
            X = log_transformation(X, self.config.xlog)

        plt.rcParams.update({"font.size": self.config.font_size})

        self.config.col_fill = color_name_to_rgb(self.config.col_fill)
        self.config.col_outline = color_name_to_rgb(self.config.col_outline)

        # Create the histogram
        plt.hist(
            X.dropna(),
            bins=self.config.bins,
            color=self.config.col_fill,
            edgecolor=self.config.col_outline,
            alpha=0.6,
        )

        plt.xlabel(self.config.xlab)
        plt.ylabel(self.config.ylab)
        plt.title(self.config.title)

        if self.config.ylog is not None:
            # Prepare y-axis label
            self.config.ylab = prepare_axis_label(self.config.ylab, self.config.ylog)

            if self.config.ylog.startswith("log2"):
                plt.yscale("log", base=2)
            elif self.config.ylog.startswith("log10"):
                plt.yscale("log", base=10)
            elif self.config.ylog.startswith("log"):
                plt.yscale("log")

        plt.tight_layout()

        return plt.gca()


@dataclass
class ComparisonHistogramConfig(PlotterConfig):
    processor_type: str = field(default="scaling", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, None], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, None], init=False, repr=False
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    r_source: str = field(
        default=(config.R_SCRIPTS / "plotting_utils.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(
        default="generate_comparison_histogram", init=False, repr=False
    )

    xlab: Optional[str] = None
    ylab: str = "Count"
    title: str = "Comparison Histogram"
    bins: int = 30
    alpha: float = 0.6
    legend_title: str = "Legend"
    legend_position: str = "top"
    subplot_title1: str = "Before"
    subplot_title2: str = "After"
    col_set: str = "Set1"
    cols: Optional[List[str]] = None
    xlog: Optional[bool] = None
    ylog: Optional[bool] = None


class ComparisonHistogramPlotter(BasePlotter):
    _config_class = ComparisonHistogramConfig
    config: ComparisonHistogramConfig

    def __init__(
        self,
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
        install_missing: bool = None,
        config: Optional[ComparisonHistogramConfig] = None,
    ):
        super().__init__(
            config=config,
            xlab=xlab,
            ylab=ylab,
            title=title,
            bins=bins,
            alpha=alpha,
            legend_title=legend_title,
            legend_position=legend_position,
            subplot_title1=subplot_title1,
            subplot_title2=subplot_title2,
            col_set=col_set,
            cols=cols,
            xlog=xlog,
            ylog=ylog,
            install_missing=install_missing,
        )
        self.plotter = None
        if self.config.r_source and self.config.main_method:
            self.r_caller = RCaller.from_script(self.config.r_source)
            enter_code = textwrap.dedent(
                """
                suppressPackageStartupMessages(require(ggplot2))
                """
            )
            exit_code = textwrap.dedent(
                """
                ggplot2::ggsave(path, results)
                """
            )
            self.plotter = self.r_caller.get_method(
                self.config.main_method, enter_code, exit_code
            )

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        if self.config.r_source and self.config.main_method:
            self.r_caller = RCaller.from_script(self.config.r_source)
            enter_code = textwrap.dedent(
                """
                suppressPackageStartupMessages(require(ggplot2))
                """
            )
            exit_code = textwrap.dedent(
                """
                ggplot2::ggsave(path, results)
                """
            )
            self.plotter = self.r_caller.get_method(
                self.config.main_method, enter_code, exit_code
            )

        return self

    def plot(
        self,
        x1,
        x2=None,
        column1: SelectedColumnTypes = None,
        column2: SelectedColumnTypes = None,
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
        path: str = None,
        device: str = "png",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(column1, column2)
        return self._plot(
            x1,
            x2,
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
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )

    def plot_arrow(self, x1, x2):
        kwargs = self.config.get_params()
        self.plotter(x1, x2, **kwargs)
