from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Type

import pyarrow as pa
from biocore.utils.import_util import is_datasets_available

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils.types import Unset

from ..plot_feature_selection import (
    FeatureSelectorPlotter,
    FeatureSelectorPlotterConfig,
)

if TYPE_CHECKING:
    pass


@dataclass
class MinPrevalencePlotterConfig(FeatureSelectorPlotterConfig):
    plot_process_desc: str = field(
        default="Plotting presence feature selection", init=False, repr=False
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
    _compare: bool = field(default=True, init=False, repr=False)
    _add_labels: bool = field(default=False, init=False, repr=False)
    processor_name: str = field(
        default="min_prevalence_feature_selector", init=False, repr=False
    )

    sample_xlab: str = "Sum of Counts"
    sample_main: str = "Sample Distribution"
    feature_xlab: str = "Sum of Counts"
    feature_main: str = "Feature Distribution"
    legend_position: str = "top"
    legend_title: str = "Max Missing Feature Selection"
    before_name: str = "Before"
    after_name: str = "After"
    xlog: str = None
    ylog: str = None
    ncol: int = 2
    include_non_zero_sum: bool = False
    non_zero_samp_xlab: str = None
    non_zero_samp_main: str = None
    non_zero_feat_xlab: str = None
    non_zero_feat_main: str = None


@dataclass
class MinPrevalencePlotterConfigForMetagenomics(MinPrevalencePlotterConfig):
    dataset_name: str = field(default="metagenomics", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )

    feature_main = "Taxa Total Abundance Distribution"
    non_zero_samp_xlab: str = "Species Richness"
    non_zero_samp_main: str = "Richness Distribution"
    non_zero_feat_xlab: str = "Species Prevalence"
    non_zero_feat_main: str = "Prevalence Across Samples"
    xlog = "log2_1p"
    include_non_zero_sum: bool = True


@dataclass
class MinPrevalencePlotterConfigForOTU(MinPrevalencePlotterConfigForMetagenomics):
    dataset_name: str = field(default="otu", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    feature_main: str = "OTU Total Abundance Distribution"
    include_non_zero_sum: bool = True


@dataclass
class MinPrevalencePlotterConfigForASV(MinPrevalencePlotterConfigForMetagenomics):
    dataset_name: str = field(default="asv", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    include_non_zero_sum: bool = True


@dataclass
class MinPrevalencePlotterConfigForGenomics(MinPrevalencePlotterConfig):
    dataset_name: str = field(default="genomics", init=False, repr=False)
    _input_feature_types: List[Type] = field(
        default_factory=lambda: [
            (get_feature("ReadCount"), get_feature("GenomicVariant")),
            (get_feature("ReadCount"), get_feature("GenomicVariant")),
        ],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalencePlotterConfigForSNP(MinPrevalencePlotterConfig):
    dataset_name: str = field(default="snp", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
        ],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalencePlotterConfigForReadCount(MinPrevalencePlotterConfig):
    dataset_name: str = field(default="read_count", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ReadCount"), get_feature("ReadCount")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ReadCount"), get_feature("ReadCount")],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalencePlotterConfigForProteomics(MinPrevalencePlotterConfig):
    dataset_name: str = field(default="proteomics", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Expression"), get_feature("Expression")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Expression"), get_feature("Expression")],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalencePlotterConfigForMaldi(MinPrevalencePlotterConfig):
    dataset_name: str = field(default="maldi", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("PeakIntensity"),
            get_feature("PeakIntensity"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("PeakIntensity"),
            get_feature("PeakIntensity"),
        ],
        init=False,
        repr=False,
    )


class MinPrevalenceFeatureSelectorPlotter(FeatureSelectorPlotter):
    config_class = MinPrevalencePlotterConfig
    config: MinPrevalencePlotterConfig

    def __init__(
        self,
        sample_xlab: str = "Sum of Counts",
        sample_main: str = "Sample Distribution",
        feature_xlab: str = "Sum of Counts",
        feature_main: str = "Feature Distribution",
        legend_position: str = "top",
        legend_title: str = "Max Missing Feature Selection",
        before_name: str = "Before",
        after_name: str = "After",
        xlog: str = None,
        ylog: str = None,
        ncol: int = 2,
        include_non_zero_sum: bool = False,
        non_zero_samp_xlab: str = None,
        non_zero_samp_main: str = None,
        non_zero_feat_xlab: str = None,
        non_zero_feat_main: str = None,
        config: MinPrevalencePlotterConfig = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            sample_xlab=sample_xlab,
            sample_main=sample_main,
            feature_xlab=feature_xlab,
            feature_main=feature_main,
            legend_position=legend_position,
            legend_title=legend_title,
            before_name=before_name,
            after_name=after_name,
            xlog=xlog,
            ylog=ylog,
            ncol=ncol,
            include_non_zero_sum=include_non_zero_sum,
            non_zero_samp_xlab=non_zero_samp_xlab,
            non_zero_samp_main=non_zero_samp_main,
            non_zero_feat_xlab=non_zero_feat_xlab,
            non_zero_feat_main=non_zero_feat_main,
            **kwargs,
        )

    def plot_pandas(self, x1, x2):
        row_sums = [pa.array(x1.sum(axis=1)), pa.array(x2.sum(axis=1))]
        col_sums = [pa.array(x1.sum(axis=0)), pa.array(x2.sum(axis=0))]
        input = [row_sums, col_sums]
        mains = [self.config.sample_main, self.config.feature_main]
        xlabs = [self.config.sample_xlab, self.config.feature_xlab]
        if self.config.include_non_zero_sum:
            non_zero_sample_sums = [
                pa.array(x1[x1 > 0].count(axis=1)),
                pa.array(x2[x2 > 0].count(axis=1)),
            ]
            non_zero_feature_sums = [
                pa.array(x1[x1 > 0].count(axis=0)),
                pa.array(x2[x2 > 0].count(axis=0)),
            ]
            mains.extend(
                [
                    self.config.non_zero_samp_main,
                    self.config.non_zero_feat_main,
                ]
            )
            xlabs.extend(
                [
                    self.config.non_zero_samp_xlab,
                    self.config.non_zero_feat_xlab,
                ]
            )
            input.extend([non_zero_sample_sums, non_zero_feature_sums])

        self.plotter(
            list_of_sums=input,
            path=self.config.path,
            xlabs=xlabs,
            mains=mains,
            ncol=self.config.ncol,
            legend_position=self.config.legend_position,
            legend_title=self.config.legend_title,
            before_name=self.config.before_name,
            after_name=self.config.after_name,
            xlog=self.config.xlog,
            ylog=self.config.ylog,
        )

    def plot(
        self,
        x1,
        x2=None,
        input_columns1: SelectedColumnTypes = None,
        input_columns2: SelectedColumnTypes = None,
        sample_xlab: str = Unset('"Sum of Counts"'),
        sample_main: str = Unset('"Sample Distribution"'),
        feature_xlab: str = Unset('"Sum of Counts"'),
        feature_main: str = Unset('"Feature Distribution"'),
        legend_position: str = Unset('"top"'),
        legend_title: str = Unset('"Max Missing Feature Selection"'),
        before_name: str = Unset('"Before"'),
        after_name: str = Unset('"After"'),
        xlog: str = Unset("None"),
        ylog: str = Unset("None"),
        ncol: int = Unset("2"),
        include_non_zero_sum: bool = Unset("False"),
        non_zero_samp_xlab: str = Unset("None"),
        non_zero_samp_main: str = Unset("None"),
        non_zero_feat_xlab: str = Unset("None"),
        non_zero_feat_main: str = Unset("None"),
        path: str = None,
        device: str = "pdf",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        if x2 is None:
            if is_datasets_available():
                from biosets import Bioset
                from datasets import Dataset as HfDataset

                if isinstance(x1, (Bioset, HfDataset)):
                    _, _, prev_replays = self._get_replays(x1)
                    if prev_replays:
                        x2 = Bioset.from_replays(prev_replays)
                    return self._plot(x2, x1)
            else:
                raise ValueError(
                    "Must provide the before and after feature selection datasets."
                )

        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns1, input_columns2
        )
        return self._plot(
            x1,
            x2,
            sample_xlab=sample_xlab,
            sample_main=sample_main,
            feature_xlab=feature_xlab,
            feature_main=feature_main,
            legend_position=legend_position,
            legend_title=legend_title,
            before_name=before_name,
            after_name=after_name,
            xlog=xlog,
            ylog=ylog,
            ncol=ncol,
            include_non_zero_sum=include_non_zero_sum,
            non_zero_samp_xlab=non_zero_samp_xlab,
            non_zero_samp_main=non_zero_samp_main,
            non_zero_feat_xlab=non_zero_feat_xlab,
            non_zero_feat_main=non_zero_feat_main,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )
