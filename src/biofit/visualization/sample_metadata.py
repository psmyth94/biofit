from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type

from biocore import DataHandler

from biofit.integration import RCaller
from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.visualization.plotting import BasePlotter, PlotterConfig

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class SampleMetadataPlotterConfig(PlotterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    r_source: str = field(
        default=(Path(__file__).parent / "plot_sample_metadata.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="plot_sample_metadata", init=False, repr=False)
    sample_metadata_columns: Optional[SelectedColumnTypes] = None
    outcome_column: Optional[SelectedColumnTypes] = None


class SampleMetadataPlotter(BasePlotter):
    _config_class = SampleMetadataPlotterConfig
    config: SampleMetadataPlotterConfig

    def __init__(
        self,
        config: Optional[SampleMetadataPlotterConfig] = None,
        sample_metadata_columns: Optional[SelectedColumnTypes] = None,
        outcome_column: Optional[SelectedColumnTypes] = None,
        install_missing: bool = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            sample_metadata_columns=sample_metadata_columns,
            outcome_column=outcome_column,
            install_missing=install_missing,
            **kwargs,
        )
        r_source = (Path(__file__).parent / "plot_sample_metadata.R").as_posix()
        r_caller = RCaller.from_script(r_source)
        self.plotter = r_caller.get_method(self.config.main_method)

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        r_source = (Path(__file__).parent / "plot_sample_metadata.R").as_posix()
        r_caller = RCaller.from_script(r_source)
        self.plotter = r_caller.get_method(self.config.main_method)
        return self

    def plot_dataset(self, X: "Dataset", y: "Dataset" = None):
        from biosets import decode

        if y is not None and isinstance(
            next(iter(y._info.features.values())), get_feature("ClassLabel")
        ):
            y = decode(y)
        current_name = next(iter(y._info.features.keys()))
        original_name = next(iter(y._info.features.values())).id or current_name
        if current_name != original_name:
            y = DataHandler.rename_column(y, current_name, original_name)
        if original_name in X.column_names:
            X = DataHandler.drop_column(X, original_name)
        self.plot_arrow(
            DataHandler.to_arrow(X), DataHandler.to_arrow(y) if y is not None else None
        )

    def plot_arrow(self, X, y):
        if self.config.outcome_column is None and y is None:
            raise ValueError(
                "No outcome column provided. Please provide the outcome data "
                "or specify the outcome column."
            )

        if self.config.outcome_column is None:
            self.config.outcome_column = list(y.column_names)[0]
        self.plotter(X, outcome=y, **self.config.get_params())

    def plot(
        self,
        sample_metadata,
        outcome_data: SelectedColumnTypes = None,
        sample_metadata_columns: Optional[SelectedColumnTypes] = None,
        outcome_column: Optional[SelectedColumnTypes] = None,
        path: str = None,
        device: str = "pdf",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        self.config._input_columns = self._set_input_columns_and_arity(
            sample_metadata_columns, outcome_column
        )
        return self._plot(
            sample_metadata,
            outcome_data,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )
