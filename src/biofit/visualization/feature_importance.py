from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np
import pandas as pd
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.integration.R.r_caller import RCaller
from biofit.processing import SelectedColumnTypes
from biofit.utils.types import Unset
from biofit.visualization.plotting import (
    BasePlotter,
    PlotterConfig,
)

if TYPE_CHECKING:
    from biosets import Dataset


@dataclass
class FeatureImportancePlotterConfig(PlotterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            None,
            None,
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES_NOT_TARGET"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            None,
            None,
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES_NOT_TARGET"),
        ],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            None,
            None,
            None,
        ],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            None,
            None,
            None,
        ],
        init=False,
        repr=False,
    )
    r_source: str = field(
        default=(Path(__file__).parent / "plot_feature_importance.R")
        .resolve()
        .as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="plot_feature_importance", init=False, repr=False)

    input_columns: SelectedColumnTypes = None
    target_columns: SelectedColumnTypes = None
    sample_metadata_columns: SelectedColumnTypes = None
    sample_column: str = None
    plot_top: int = 15
    feature_meta_name: str = None
    feature_column: str = None
    cols: List[str] = None
    colHeat: List[str] = None
    dat_log: str = field(default=None, init=True, repr=False)
    show_column_names: bool = False
    scale_legend_title: str = "Value"
    column_title: str = "Samples"
    row_title: str = "Features"
    plot_title: str = "Values"

    def __post_init__(self):
        if self.dat_log == ["log2_1p", "log2"]:
            self.plot_title = "log2\n" + self.plot_title
        elif self.dat_log == ["log10_1p", "log10"]:
            self.plot_title = "log10\n" + self.plot_title


@dataclass
class FeatureImportancePlotterConfigForMetagenomics(FeatureImportancePlotterConfig):
    dat_log: str = "log2_1p"
    plot_title: str = "Abundance"
    row_title: str = "Taxa"


@dataclass
class FeatureImportancePlotterConfigForOTU(
    FeatureImportancePlotterConfigForMetagenomics
):
    row_title: str = "OTUs"


class FeatureImportancePlotter(BasePlotter):
    _config_class = FeatureImportancePlotterConfig
    config: FeatureImportancePlotterConfig

    def __init__(
        self,
        input_columns: SelectedColumnTypes = None,
        target_columns: SelectedColumnTypes = None,
        sample_metadata_columns: SelectedColumnTypes = None,
        sample_column: str = None,
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
        path: Optional[str] = None,
        install_missing: bool = None,
        config: Optional[FeatureImportancePlotterConfig] = None,
    ):
        super().__init__(
            plot_top=plot_top,
            feature_meta_name=feature_meta_name,
            feature_column=feature_column,
            cols=cols,
            colHeat=colHeat,
            dat_log=dat_log,
            show_column_names=show_column_names,
            scale_legend_title=scale_legend_title,
            column_title=column_title,
            row_title=row_title,
            plot_title=plot_title,
            config=config,
            install_missing=install_missing,
        )
        self.r_caller = RCaller.from_script(self.config.r_source)
        self.r_caller.verify_r_dependencies(
            bioconductor_dependencies=["ComplexHeatmap"],
            install_missing=install_missing,
        )
        self.plotter = self.r_caller.get_method(self.config.main_method)

    def plot_dataset(
        self,
        X: "Dataset",
        feature_importances: "Dataset",
        y: "Dataset",
        sample_metadata: "Dataset" = None,
        feature_metadata: dict = None,
    ):
        from biosets import decode, get_feature_metadata, get_sample_col_name

        if feature_metadata is None:
            feature_metadata = get_feature_metadata(X)

        if self.config.sample_column is None:
            self.config.sample_column = (
                get_sample_col_name(sample_metadata)
                if sample_metadata is not None
                else get_sample_col_name(X)
            )

        self.plot_pandas(
            X=DataHandler.to_pandas(X),
            feature_importances=DataHandler.to_pandas(feature_importances),
            y=DataHandler.to_pandas(decode(y)) if y is not None else None,
            sample_metadata=DataHandler.to_pandas(sample_metadata)
            if sample_metadata is not None
            else None,
            feature_metadata=feature_metadata,
        )

    def plot_pandas(
        self,
        X: pd.DataFrame,
        feature_importances: pd.DataFrame,
        y: pd.DataFrame,
        sample_metadata: pd.DataFrame = None,
        feature_metadata: dict = None,
    ):
        if self.config.dat_log == "log2_1p":
            X = np.log2(X + 1)
        elif self.config.dat_log == "log2":
            X = np.log2(X)
        elif self.config.dat_log == "log10_1p":
            X = np.log10(X + 1)
        elif self.config.dat_log == "log10":
            X = np.log10(X)

        if feature_importances is None:
            raise ValueError("Please provide feature importances.")

        self.config.feature_column = (
            self.config.feature_column or feature_importances.columns[0]
        )
        if self.config.feature_column not in feature_importances.columns:
            raise ValueError(
                f"Feature column '{self.config.feature_column}' not found in feature "
                "importances. Please provide the column name found in both "
                "feature importances and feature metadata (if provided)."
            )

        feat_import_cols = [
            c for c in feature_importances.columns if c != self.config.feature_column
        ]

        if (
            isinstance(feature_metadata, dict)
            and feature_metadata is not None
            and len(feature_metadata) > 0
        ):
            feature_metadata = pd.DataFrame(
                list(feature_metadata.values()), index=list(feature_metadata.keys())
            )
            feature_metadata = feature_metadata.reset_index(
                names=[self.config.feature_column]
            )

        if len(feat_import_cols) > 1:
            medians = feature_importances.loc[:, feat_import_cols].median(axis=1)
            sorted_inds = np.argsort(np.abs(medians))[::-1]
            feature_importances = feature_importances.iloc[sorted_inds, :].head(
                self.config.plot_top
            )
        X = DataHandler.to_arrow(X, preserve_index=False)
        feature_importances = DataHandler.to_arrow(
            feature_importances, preserve_index=False
        )
        if y is not None:
            y = DataHandler.to_arrow(y, preserve_index=False)
        if feature_metadata is not None and len(feature_metadata) > 0:
            feature_metadata = DataHandler.to_arrow(
                feature_metadata, preserve_index=False
            )
        if sample_metadata is not None:
            sample_metadata = DataHandler.to_arrow(
                sample_metadata, preserve_index=False
            )

        self._plotter(
            X,
            y=y,
            feature_importances=feature_importances,
            sample_metadata=sample_metadata,
            feature_metadata=feature_metadata,
        )

    def _plotter(
        self,
        X,
        y,
        feature_importances,
        sample_metadata,
        feature_metadata,
    ):
        params = self.config.get_params()
        if feature_metadata is not None:
            feature_metadata_columns = DataHandler.get_column_names(feature_metadata)
            feature_meta_name = params.get("feature_meta_name")
            if feature_meta_name is not None:
                if isinstance(feature_meta_name, (str, int)):
                    feature_meta_name = [feature_meta_name]
                missing_cols = set(feature_meta_name) - set(feature_metadata_columns)
                if missing_cols:
                    raise ValueError(
                        f"Feature metadata columns {list(missing_cols)} not found in "
                        "feature metadata."
                    )

        self.plotter(
            X,
            y=y,
            feature_importances=feature_importances,
            sample_metadata=sample_metadata,
            feature_metadata=feature_metadata,
            **params,
        )

    def plot(
        self,
        X,
        feature_importances,
        y=None,
        sample_metadata=None,
        feature_metadata: dict = None,
        input_columns: SelectedColumnTypes = None,
        target_columns: SelectedColumnTypes = None,
        sample_metadata_columns: SelectedColumnTypes = None,
        sample_column: str = None,
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
        path: str = None,
        device: str = "pdf",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show=True,
    ):
        # feature_importances are not selected by columns so its None
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, None, target_columns, sample_metadata_columns
        )
        if (
            feature_importances is not None
            and DataHandler.get_shape(feature_importances)[0] == 0
        ):
            raise ValueError("Feature importances is empty.")
        self._plot(
            X,
            feature_importances,
            y,
            sample_metadata,
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
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
        )
