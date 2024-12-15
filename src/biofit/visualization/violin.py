import textwrap
from dataclasses import dataclass, field
from typing import List, Type

import biofit.config as config
from biofit.integration.biosets import get_feature
from biofit.integration.R import RCaller
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils.types import Unset
from biofit.visualization.plotting import BasePlotter, PlotterConfig


@dataclass
class ViolinConfig(PlotterConfig):
    processor_type: str = field(default="scaling", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )

    r_source: str = field(
        default=(config.R_SCRIPTS / "plotting_utils.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="generate_violin", init=False, repr=False)

    column: str = None
    label_name: str = "labels"
    xlab: str = "Labels"
    ylab: str = "Value"


class ViolinPlotter(BasePlotter):
    _config_class = ViolinConfig
    config: ViolinConfig

    def __init__(
        self,
        column: str = None,
        label_name: str = None,
        xlab: str = Unset('"Labels"'),
        ylab: str = Unset('"Value"'),
        install_missing: bool = None,
        config=None,
    ):
        super().__init__(
            config=config, xlab=xlab, ylab=ylab, install_missing=install_missing
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
        column: SelectedColumnTypes = None,
        label_name: SelectedColumnTypes = None,
        xlab: str = Unset('"Labels"'),
        ylab: str = Unset('"Value"'),
        path: str = None,
        device: str = "png",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(
            column, label_name
        )
        return self._plot(
            x,
            y,
            xlab=xlab,
            ylab=ylab,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )

    def plot_arrow(self, x, y=None):
        self.plotter(
            x,
            y,
            **self.config.get_params(),
        )
