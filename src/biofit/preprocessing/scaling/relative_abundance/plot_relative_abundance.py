from dataclasses import dataclass, field
from typing import List, Optional, Type

from biocore.utils.py_util import is_bioset

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils.types import Unset

from ..plot_scaling import ScalerPlotter, ScalerPlotterConfig


@dataclass
class RelativeAbundancePlotterConfig(ScalerPlotterConfig):
    processor_name: str = field(default="relative_abundance", init=False, repr=False)
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
    before_title: str = "Before Relative Abundance"
    after_title: str = "After Relative Abundance"
    legend_position: str = "top"
    add_box: bool = True
    horizontal_plot: bool = False
    order: bool = False
    col_set: str = "Set1"
    cols: Optional[str] = None
    log_num: Optional[str] = None
    show_outliers: bool = True


@dataclass
class RelativeAbundancePlotterConfigForMetagenomics(RelativeAbundancePlotterConfig):
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
    experiment_name: str = field(default="metagenomics", init=False, repr=False)
    log_num: Optional[str] = "log2_1p"


@dataclass
class RelativeAbundancePlotterConfigForOTU(RelativeAbundancePlotterConfig):
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
    experiment_name: str = field(default="otu", init=False, repr=False)
    ylab: Optional[str] = "OTU Abundance"
    log_num: Optional[str] = "log10_1p"


@dataclass
class RelativeAbundancePlotterConfigForASV(RelativeAbundancePlotterConfig):
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
    experiment_name: str = field(default="asv", init=False, repr=False)


@dataclass
class RelativeAbundancePlotterConfigForSNP(RelativeAbundancePlotterConfig):
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
    experiment_name: str = field(default="snp", init=False, repr=False)


@dataclass
class RelativeAbundancePlotterConfigForMaldi(RelativeAbundancePlotterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("PeakIntensity"),
            get_feature("PeakIntensity"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("PeakIntensity"),
            get_feature("PeakIntensity"),
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    experiment_name: str = field(default="maldi", init=False, repr=False)
    ylab: Optional[str] = "Peak Intensity"


class RelativeAbundancePlotter(ScalerPlotter):
    _config_class = RelativeAbundancePlotterConfig
    config: RelativeAbundancePlotterConfig

    def __init__(
        self,
        input_columns1: SelectedColumnTypes = None,
        input_columns2: SelectedColumnTypes = None,
        label_name1: Optional[str] = None,
        label_name2: Optional[str] = None,
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
        config: RelativeAbundancePlotterConfig = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            input_columns1=input_columns1,
            input_columns2=input_columns2,
            label_name1=label_name1,
            label_name2=label_name2,
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
        device: str = "png",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        if is_bioset(x1):
            from biosets import get_target

            if y1 is None:
                y1 = get_target(x1)

        if is_bioset(x2):
            from biosets import get_target

            if y2 is None:
                y2 = get_target(x2)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            raise ValueError("Must provide the before and after normalization.")
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns1, input_columns2, label_name1, label_name2
        )
        return self._plot(
            x1,
            x2,
            y1,
            y2,
            input_columns1=input_columns1,
            input_columns2=input_columns2,
            label_name1=label_name1,
            label_name2=label_name2,
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
