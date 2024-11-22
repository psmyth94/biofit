from dataclasses import dataclass, field
from typing import List, Type

import pyarrow as pa

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils.types import Unset

from ..plot_filtering import (
    SampleFilterPlotter,
    SampleFilterPlotterConfig,
)


@dataclass
class MinPrevalenceRowPlotterConfig(SampleFilterPlotterConfig):
    plot_process_desc: str = field(
        default="Plotting presence feature selection", init=False, repr=False
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
    processor_name: str = field(
        default="min_prevalence_sample_filter", init=False, repr=False
    )

    sample_xlab: str = "Sum of Counts"
    sample_main: str = "Sample Distribution"
    feature_xlab: str = "Sum of Counts"
    feature_main: str = "Feature Distribution"
    legend_position: str = "top"
    legend_title: str = "Min Prevalence Sample SampleFiltering"
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
class MinPrevalenceRowPlotterConfigForMetagenomics(MinPrevalenceRowPlotterConfig):
    experiment_name: str = field(default="metagenomics", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            [get_feature("Abundance"), get_feature("ReadCount")],
            [get_feature("Abundance"), get_feature("ReadCount")],
        ],
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
class MinPrevalenceRowPlotterConfigForOTU(MinPrevalenceRowPlotterConfigForMetagenomics):
    experiment_name: str = field(default="otu", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    feature_main: str = "OTU Total Abundance Distribution"
    include_non_zero_sum: bool = True


@dataclass
class MinPrevalenceRowPlotterConfigForASV(MinPrevalenceRowPlotterConfigForMetagenomics):
    experiment_name: str = field(default="asv", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance"), get_feature("Abundance")],
        init=False,
        repr=False,
    )
    include_non_zero_sum: bool = True


@dataclass
class MinPrevalenceRowPlotterConfigForGenomics(MinPrevalenceRowPlotterConfig):
    experiment_name: str = field(default="genomics", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            (get_feature("ReadCount"), get_feature("GenomicVariant")),
            (get_feature("ReadCount"), get_feature("GenomicVariant")),
        ],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalenceRowPlotterConfigForSNP(MinPrevalenceRowPlotterConfig):
    experiment_name: str = field(default="snp", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("GenomicVariant"),
        ],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalenceRowPlotterConfigForMaldi(MinPrevalenceRowPlotterConfig):
    experiment_name: str = field(default="maldi", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("PeakIntensity"),
            get_feature("PeakIntensity"),
        ],
        init=False,
        repr=False,
    )
    feature_main: str = "Peak Total Intensity Distribution"
    non_zero_samp_xlab: str = "Peak Richness"
    non_zero_samp_main: str = "Richness Distribution"
    non_zero_feat_xlab: str = "Peak Prevalence"
    non_zero_feat_main: str = "Prevalence Across Samples"
    xlog = "log2_1p"
    include_non_zero_sum: bool = True


@dataclass
class MinPrevalenceRowPlotterConfigForReadCount(MinPrevalenceRowPlotterConfig):
    experiment_name: str = field(default="read_count", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ReadCount"), get_feature("ReadCount")],
        init=False,
        repr=False,
    )


@dataclass
class MinPrevalenceRowPlotterConfigForProteomics(MinPrevalenceRowPlotterConfig):
    experiment_name: str = field(default="proteomics", init=False, repr=False)
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("PeakIntensity"),
            get_feature("PeakIntensity"),
        ],
        init=False,
        repr=False,
    )


class MinPrevalenceRowPlotter(SampleFilterPlotter):
    _config_class = MinPrevalenceRowPlotterConfig
    config: MinPrevalenceRowPlotterConfig

    def __init__(
        self,
        sample_xlab: str = Unset('"Sum of Counts"'),
        sample_main: str = Unset('"Sample Distribution"'),
        feature_xlab: str = Unset('"Sum of Counts"'),
        feature_main: str = Unset('"Feature Distribution"'),
        legend_position: str = Unset('"top"'),
        legend_title: str = Unset('"Min Prevalence Sample SampleFiltering"'),
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
        config: MinPrevalenceRowPlotterConfig = None,
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
            xlabs=xlabs,
            mains=mains,
            **self.config.get_params(),
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
        legend_title: str = Unset('"Min Prevalence Sample SampleFiltering"'),
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
