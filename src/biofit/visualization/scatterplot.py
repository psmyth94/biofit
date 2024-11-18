import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Type

import biofit.config as config
from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils.types import Unset
from biofit.visualization.plotting import BasePlotter, PlotterConfig


@dataclass
class ScatterPlotConfig(PlotterConfig):
    processor_type: str = field(default="scaling", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None, None],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None, None],
        init=False,
        repr=False,
    )
    r_source: str = field(
        default=(config.R_SCRIPTS / "plotting_utils.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="generate_scatterplot", init=False, repr=False)

    groupby: str = None
    xdata: str = None
    ydata: str = None
    xlab: str = None
    ylab: str = None
    title: str = "Scatterplot"
    alpha: str = 1
    col_set: str = "Set1"
    cols: List[str] = None
    xlog: str = None
    ylog: str = None


class ScatterPlotter(BasePlotter):
    _config_class = ScatterPlotConfig
    config: ScatterPlotConfig

    def __init__(
        self,
        groupby: str = None,
        xdata: str = None,
        ydata: str = None,
        xlab: str = Unset("None"),
        ylab: str = Unset("None"),
        title: str = Unset('"Scatterplot"'),
        alpha: str = Unset("1"),
        col_set: str = Unset('"Set1"'),
        cols: List[str] = Unset("None"),
        xlog: str = Unset("None"),
        ylog: str = Unset("None"),
        install_missing: bool = None,
        config: Optional[ScatterPlotConfig] = None,
    ):
        super().__init__(
            config=config,
            xlab=xlab,
            ylab=ylab,
            title=title,
            alpha=alpha,
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
        path: str = None,
        device: str = "pdf",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(
            xdata, ydata, groupby
        )
        return self._plot(
            x,
            y,
            group,
            xlab=xlab,
            ylab=ylab,
            title=title,
            alpha=alpha,
            col_set=col_set,
            cols=cols,
            xlog=xlog,
            ylog=ylog,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )

    def plot_dataset(self, x, y=None, group=None):
        from biosets import Bioset, decode

        if isinstance(group, Bioset):
            group = decode(group)

        return self.plot_arrow(
            x._data.table,
            y._data.table if y else None,
            group._data.table if group else None,
        )

    def plot_arrow(self, x, y=None, group=None):
        kwargs = self.config.get_params()
        context_kwargs = {
            "path": kwargs.pop("path", None),
        }
        self.plotter(x, y, group, context_kwargs=context_kwargs, **kwargs)
