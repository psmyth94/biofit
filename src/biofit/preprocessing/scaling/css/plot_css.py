from dataclasses import dataclass, field
from typing import List, Optional, Type

from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biocore.utils.py_util import is_bioset
from biofit.utils.types import Unset

from ..plot_scaling import ScalerPlotter, ScalerPlotterConfig


@dataclass
class CumulativeSumScalerPlotterConfig(ScalerPlotterConfig):
    processor_name: str = field(default="css", init=False, repr=False)
    _compare: bool = field(default=True, init=False, repr=False)
    _transoform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            None,
            None,
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )

    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES"),
            None,
            None,
        ],
        init=False,
        repr=False,
    )

    input_columns1: Optional[str] = None
    input_columns2: Optional[str] = None
    label_name1: Optional[str] = None
    label_name2: Optional[str] = None
    ylab: Optional[str] = None
    xlab: Optional[str] = None
    ylim: Optional[list] = None
    before_title: str = "Before CSS"
    after_title: str = "After CSS"
    legend_position: str = "top"
    add_box: bool = True
    horizontal_plot: bool = False
    order: bool = False
    col_set: str = "Set1"
    cols: Optional[str] = None
    log_num: Optional[str] = None
    show_outliers: bool = True


@dataclass
class CumulativeSumScalerPlotterConfigForMetagenomics(CumulativeSumScalerPlotterConfig):
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            (get_feature("Abundance"), get_feature("ReadCount")),
            (get_feature("Abundance"), get_feature("ReadCount")),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="metagenomics", init=False, repr=False)
    log_num: Optional[str] = "log2_1p"


@dataclass
class CumulativeSumScalerPlotterConfigForOTU(CumulativeSumScalerPlotterConfig):
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("Abundance"),
            get_feature("Abundance"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="otu", init=False, repr=False)
    ylab: Optional[str] = "OTU Abundance"
    log_num: Optional[str] = "log10_1p"


@dataclass
class CumulativeSumScalerPlotterConfigForASV(CumulativeSumScalerPlotterConfig):
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("Abundance"),
            get_feature("Abundance"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="asv", init=False, repr=False)


@dataclass
class CumulativeSumScalerPlotterConfigForGenomics(CumulativeSumScalerPlotterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="genomics", init=False, repr=False)


@dataclass
class CumulativeSumScalerPlotterConfigForSNP(CumulativeSumScalerPlotterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="snp", init=False, repr=False)


@dataclass
class CumulativeSumScalerPlotterConfigForReadCount(CumulativeSumScalerPlotterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("ReadCount"),
            get_feature("ReadCount"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("ReadCount"),
            get_feature("ReadCount"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="read_count", init=False, repr=False)


class CumulativeSumScalerPlotter(ScalerPlotter):
    config_class = CumulativeSumScalerPlotterConfig
    config: CumulativeSumScalerPlotterConfig

    def __init__(
        self,
        ylab: Optional[str] = Unset("None"),
        xlab: Optional[str] = Unset("None"),
        ylim: Optional[list] = Unset("None"),
        title: str = Unset('"Violin Plot"'),
        legend_position: str = Unset('"top"'),
        add_box: bool = Unset("True"),
        horizontal_plot: bool = Unset("False"),
        order: bool = Unset("False"),
        col_set: str = Unset('"Set1"'),
        cols: Optional[str] = Unset("None"),
        log_num: Optional[int] = Unset("None"),
        show_outliers: bool = Unset("True"),
        config: CumulativeSumScalerPlotterConfig = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            ylab=ylab,
            xlab=xlab,
            ylim=ylim,
            title=title,
            legend_position=legend_position,
            add_box=add_box,
            horizontal_plot=horizontal_plot,
            order=order,
            col_set=col_set,
            cols=cols,
            log_num=log_num,
            show_outliers=show_outliers,
            **kwargs,
        )

    def plot(
        self,
        x1,
        x2=None,
        y1=None,
        y2=None,
        input_columns1: SelectedColumnTypes = None,
        input_columns2: SelectedColumnTypes = None,
        label_name1: SelectedColumnTypes = None,
        label_name2: SelectedColumnTypes = None,
        ylab: Optional[str] = Unset("None"),
        xlab: Optional[str] = Unset("None"),
        ylim: Optional[list] = Unset("None"),
        title: str = Unset('"Violin Plot"'),
        legend_position: str = Unset('"top"'),
        add_box: bool = Unset("True"),
        horizontal_plot: bool = Unset("False"),
        order: bool = Unset("False"),
        col_set: str = Unset('"Set1"'),
        cols: Optional[str] = Unset("None"),
        log_num: Optional[int] = Unset("None"),
        show_outliers: bool = Unset("True"),
        path: str = None,
        device: str = "pdf",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        if is_bioset(x2):
            from biosets import get_target

            if y2 is None:
                y2 = get_target(x2)
        if x1 is None or x2 is None:
            raise ValueError("Must provide the before and after normalization.")
        if y1 is not None and y2 is None:
            raise ValueError("Must provide the target for the after normalization.")
        if y1 is None:
            if label_name1 is None:
                raise ValueError(
                    "Must provide the target for the before normalization."
                )
            y1 = DataHandler.select_columns(x1, label_name1)
            x1 = DataHandler.drop_columns(x1, label_name1)
        if y2 is None:
            if label_name2 is None:
                raise ValueError("Must provide the target for the after normalization.")
            y2 = DataHandler.select_columns(x2, label_name2)
            x2 = DataHandler.drop_columns(x2, label_name2)

        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns1, input_columns2, label_name1, label_name2
        )
        return self._plot(
            x1,
            x2,
            y1,
            y2,
            ylab=ylab,
            xlab=xlab,
            ylim=ylim,
            title=title,
            legend_position=legend_position,
            add_box=add_box,
            horizontal_plot=horizontal_plot,
            order=order,
            col_set=col_set,
            cols=cols,
            log_num=log_num,
            show_outliers=show_outliers,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )
