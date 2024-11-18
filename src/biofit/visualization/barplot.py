import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Type

from biocore import DataHandler
from biocore.utils.py_util import is_bioset

import biofit.config as config
from biofit.integration.biosets import get_feature
from biofit.integration.R import RCaller
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils.types import Unset
from biofit.visualization.plotting import BasePlotter, PlotterConfig


@dataclass
class BarPlotConfig(PlotterConfig):
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

    processor_type: str = field(default="scaling", init=False, repr=False)
    r_source: str = field(
        default=(config.R_SCRIPTS / "plotting_utils.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="generate_barplot", init=False, repr=False)

    label_name: str = None
    value_name: Optional[str] = None
    groupby: Optional[str] = None
    xlab: Optional[str] = None
    ylab: Optional[str] = None
    title: str = "Bar Plot"
    col_set: str = "Set1"
    col_outline: str = "grey30"
    col_labels: str = "black"
    cols: Optional[str] = None
    prop: bool = False
    add_count_lab: bool = True
    vars_as_entered: bool = False
    legend_position: str = "top"
    font_size: float = 3.25


class BarPlotter(BasePlotter):
    _config_class = BarPlotConfig
    config: BarPlotConfig

    def __init__(
        self,
        xlab: Optional[str] = None,
        ylab: Optional[str] = None,
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
        path=None,
        config: Optional[BarPlotConfig] = None,
    ):
        super().__init__(
            config=config,
            xlab=xlab,
            ylab=ylab,
            title=title,
            col_set=col_set,
            col_labels=col_labels,
            cols=cols,
            prop=prop,
            add_count_lab=add_count_lab,
            vars_as_entered=vars_as_entered,
            legend_position=legend_position,
            font_size=font_size,
            path=path,
        )

        self = self.__post_init__()

    def __post_init__(self):
        if self.config.r_source and self.config.main_method:
            self.r_caller = RCaller.from_script(self.config.r_source)
            self.r_caller.verify_r_dependencies()
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

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self = self.__post_init__()
        return self

    def plot(
        self,
        x,
        y,
        group,
        label_name: str = None,
        value_name: Optional[str] = None,
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
        path: str = None,
        device: str = "pdf",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(
            value_name, label_name, groupby
        )
        return self._plot(
            x,
            y,
            group,
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
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )

    def plot_dataset(self, x, y, group):
        if is_bioset(x):
            from biosets import decode

            x = decode(x)
        if is_bioset(group):
            group = decode(group)

        return self.plot_arrow(
            DataHandler.to_arrow(x),
            DataHandler.to_arrow(y) if y is not None else None,
            DataHandler.to_arrow(group) if group is not None else None,
        )

    def plot_arrow(self, x, y=None, group=None):
        kwargs = self.config.get_params()
        self.plotter(x, y, group, **kwargs)
