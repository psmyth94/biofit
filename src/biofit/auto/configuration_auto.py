import importlib
import re
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Union

from biocore.utils.import_util import (
    is_biosets_available,
    is_transformers_available,
)
from biocore.utils.inspect import get_kwargs
from biocore.utils.py_util import is_bioset

from biofit.processing import ProcessorConfig

if TYPE_CHECKING:
    from biosets import Bioset


PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStat"),
        ("row_sum", "RowSumStat"),
        ("correlation", "CorrelationStat"),
        ("distance", "DistanceStat"),
        ("row_missingness", "RowMissingnessStat"),
        ("row_mean", "RowMeanStat"),
        ("col_missingness", "ColumnMissingnessStat"),
        ("col_mean", "ColumnMeanStat"),
        ("lightgbm", "LightGBMModel"),
        ("lasso", "LassoModel"),
        ("random_forest", "RandomForestModel"),
        ("logistic_regression", "LogisticRegressionModel"),
        ("pcoa", "PCoAFeatureExtractor"),
        ("pca", "PCAFeatureExtractor"),
        ("label_binarizing", "LabelBinarizer"),
        ("label_encoding", "LabelEncoder"),
        ("upsampling", "UpSampler"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelector"),
        ("log", "LogTransformer"),
        ("relative_abundance", "RelativeAbundanceScaler"),
        ("tmm", "TMMScaler"),
        ("clr", "CLRScaler"),
        ("css", "CumulativeSumScaler"),
        ("min_prevalence_sample_filter", "MinPrevalenceSampleFilter"),
        ("missing_labels", "MissingLabelsSampleFilter"),
        ("row_abundance", "AbundanceSampleFilter"),
    ]
)

PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotter"),
        ("pca", "PCAFeatureExtractorPlotter"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorPlotter"),
        ("relative_abundance", "RelativeAbundancePlotter"),
        ("css", "CumulativeSumScalerPlotter"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotter"),
        ("row_abundance", "AbundanceSampleFilterPlotter"),
    ]
)

CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

PLOTTER_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfig"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfig"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfig"),
    ]
)

PROCESSOR_CATEGORY_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "stat"),
        ("row_sum", "stat"),
        ("correlation", "stat"),
        ("distance", "stat"),
        ("row_missingness", "stat"),
        ("row_mean", "stat"),
        ("col_missingness", "stat"),
        ("col_mean", "stat"),
        ("lightgbm", "models"),
        ("lasso", "models"),
        ("random_forest", "models"),
        ("logistic_regression", "models"),
        ("pcoa", "preprocessing"),
        ("pca", "preprocessing"),
        ("label_binarizing", "preprocessing"),
        ("label_encoding", "preprocessing"),
        ("upsampling", "preprocessing"),
        ("min_prevalence_feature_selector", "preprocessing"),
        ("log", "preprocessing"),
        ("relative_abundance", "preprocessing"),
        ("tmm", "preprocessing"),
        ("clr", "preprocessing"),
        ("css", "preprocessing"),
        ("min_prevalence_sample_filter", "preprocessing"),
        ("missing_labels", "preprocessing"),
        ("row_abundance", "preprocessing"),
    ]
)

PROCESSOR_TYPE_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "feature_extraction"),
        ("pca", "feature_extraction"),
        ("label_binarizing", "encoding"),
        ("label_encoding", "encoding"),
        ("upsampling", "resampling"),
        ("min_prevalence_feature_selector", "feature_selection"),
        ("log", "transformation"),
        ("relative_abundance", "scaling"),
        ("tmm", "scaling"),
        ("clr", "scaling"),
        ("css", "scaling"),
        ("min_prevalence_sample_filter", "filtering"),
        ("missing_labels", "filtering"),
        ("row_abundance", "filtering"),
    ]
)

RNA_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

KMER_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

BIODATA_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

GENOMICS_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

SNP_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfigForSNP"),
        ("row_missingness", "RowMissingnessStatConfigForSNP"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfigForSNP"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfigForSNP"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfigForSNP"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfigForSNP"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

OTU_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfigForOTU"),
        ("row_sum", "RowSumStatConfigForOTU"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfigForOTU"),
        ("row_missingness", "RowMissingnessStatConfigForOTU"),
        ("row_mean", "RowMeanStatConfigForOTU"),
        ("col_missingness", "ColumnMissingnessStatConfigForOTU"),
        ("col_mean", "ColumnMeanStatConfigForOTU"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfigForOTU"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfigForOTU"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfigForOTU"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfigForOTU"),
        ("log", "LogTransformerConfigForOTU"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfigForOTU"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfigForOTU"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfigForOTU"),
    ]
)

MALDI_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfigForMaldi"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

MS2_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

PROTEOMICS_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

METAGENOMICS_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfigForMetagenomics"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfigForMetagenomics"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfigForMetagenomics"),
        (
            "min_prevalence_feature_selector",
            "MinPrevalenceFeatureSelectorConfigForMetagenomics",
        ),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfigForMetagenomics"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

MS1_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

METABOLOMICS_MAPPING_NAMES = OrderedDict(
    [
        ("col_sum", "ColumnSumStatConfig"),
        ("row_sum", "RowSumStatConfig"),
        ("correlation", "CorrelationStatConfig"),
        ("distance", "DistanceStatConfig"),
        ("row_missingness", "RowMissingnessStatConfig"),
        ("row_mean", "RowMeanStatConfig"),
        ("col_missingness", "ColumnMissingnessStatConfig"),
        ("col_mean", "ColumnMeanStatConfig"),
        ("lightgbm", "LightGBMConfig"),
        ("lasso", "LassoConfig"),
        ("random_forest", "RandomForestConfig"),
        ("logistic_regression", "LogisticRegressionConfig"),
        ("pcoa", "PCoAFeatureExtractorConfig"),
        ("pca", "PCAFeatureExtractorConfig"),
        ("label_binarizing", "LabelBinarizerConfig"),
        ("label_encoding", "LabelEncoderConfig"),
        ("upsampling", "UpSamplerConfig"),
        ("min_prevalence_feature_selector", "MinPrevalenceFeatureSelectorConfig"),
        ("log", "LogTransformerConfig"),
        ("relative_abundance", "RelativeAbundanceScalerConfig"),
        ("tmm", "TMMScalerConfig"),
        ("clr", "CLRScalerConfig"),
        ("css", "CumulativeSumScalerConfig"),
        ("imputation", "ImputationConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowSampleFilterConfig"),
        ("missing_labels", "MissingLabelsSampleFilterConfig"),
        ("row_abundance", "AbundanceSampleFilterConfig"),
    ]
)

RNA_SEQ_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfig"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfig"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfig"),
    ]
)

KMER_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForGenomics"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfigForGenomics"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForGenomics"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForGenomics"),
    ]
)

BIODATA_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfig"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfig"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfig"),
    ]
)

GENOMICS_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForGenomics"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfigForGenomics"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForGenomics"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForGenomics"),
    ]
)

SNP_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForSNP"),
        ("relative_abundance", "RelativeAbundancePlotterConfigForSNP"),
        ("css", "CumulativeSumScalerPlotterConfigForSNP"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForSNP"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForSNP"),
    ]
)

OTU_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForOTU"),
        ("relative_abundance", "RelativeAbundancePlotterConfigForOTU"),
        ("css", "CumulativeSumScalerPlotterConfigForOTU"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForOTU"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForOTU"),
    ]
)

MALDI_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForMaldi"),
        ("relative_abundance", "RelativeAbundancePlotterConfigForMaldi"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForMaldi"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForProteomics"),
    ]
)

MS2_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForProteomics"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForProteomics"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForProteomics"),
    ]
)

PROTEOMICS_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForProteomics"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForProteomics"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForProteomics"),
    ]
)

METAGENOMICS_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        (
            "min_prevalence_feature_selector",
            "MinPrevalencePlotterConfigForMetagenomics",
        ),
        ("relative_abundance", "RelativeAbundancePlotterConfigForMetagenomics"),
        ("css", "CumulativeSumScalerPlotterConfigForMetagenomics"),
        (
            "min_prevalence_sample_filter",
            "MinPrevalenceRowPlotterConfigForMetagenomics",
        ),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForMetagenomics"),
    ]
)

MS1_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfigForProteomics"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfigForProteomics"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfigForProteomics"),
    ]
)

METABOLOMICS_PLOTTER_MAPPING_NAMES = OrderedDict(
    [
        ("pcoa", "PCoAFeatureExtractorPlotterConfig"),
        ("pca", "PCAFeatureExtractorPlotterConfig"),
        ("min_prevalence_feature_selector", "MinPrevalencePlotterConfig"),
        ("relative_abundance", "RelativeAbundancePlotterConfig"),
        ("css", "CumulativeSumScalerPlotterConfig"),
        ("min_prevalence_sample_filter", "MinPrevalenceRowPlotterConfig"),
        ("row_abundance", "AbundanceSampleFilterPlotterConfig"),
    ]
)


def config_class_to_processor_name(config, mapping_names, mapping):
    """Converts a config class name to the corresponding model type"""
    for key, cls in mapping_names.items():
        if cls == config:
            return key
    for key, cls in mapping._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


def get_experiment_name(dataset: Union[str, "Bioset"]):
    if is_biosets_available():
        from biosets.packaged_modules import EXPERIMENT_TYPE_ALIAS
    else:
        EXPERIMENT_TYPE_ALIAS = {}
    if is_bioset(dataset):
        dataset = dataset._info.builder_name
    dataset = EXPERIMENT_TYPE_ALIAS.get(dataset, dataset)
    return dataset


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key: str):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = key.replace("-", "_")
        processor_category = PROCESSOR_CATEGORY_MAPPING_NAMES.get(key, "models")
        processor_type = PROCESSOR_TYPE_MAPPING_NAMES.get(key, None)
        try:
            if module_name not in self._modules:
                package = (
                    f"biofit.{processor_category}"
                    if not processor_type
                    else f"biofit.{processor_category}.{processor_type}"
                )
                self._modules[module_name] = importlib.import_module(
                    f".{module_name}", package
                )
        except ImportError:
            if is_transformers_available() and processor_category == "models":
                from transformers.models.auto.configuration_auto import (
                    model_type_to_module_name,
                )

                module_name = model_type_to_module_name(key)
                if module_name not in self._modules:
                    self._modules[module_name] = importlib.import_module(
                        f".{module_name}", "transformers.models"
                    )
            else:
                raise
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)
        try:
            genomicsml_module = importlib.import_module("biofit")
            return getattr(genomicsml_module, value)
        except AttributeError:
            if is_transformers_available():
                transformers_module = importlib.import_module("transformers")
                return getattr(transformers_module, value)
            raise

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(
            self._extra_content.values()
        )

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(
            self._extra_content.items()
        )

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and (not exist_ok):
            raise ValueError(
                f"'{key}' is already used by a Transformers config, pick another name."
            )
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
PLOTTER_CONFIG_MAPPING = _LazyConfigMapping(PLOTTER_CONFIG_MAPPING_NAMES)
RNA_SEQ_MAPPING = _LazyConfigMapping(RNA_SEQ_MAPPING_NAMES)
KMER_MAPPING = _LazyConfigMapping(KMER_MAPPING_NAMES)
BIODATA_MAPPING = _LazyConfigMapping(BIODATA_MAPPING_NAMES)
GENOMICS_MAPPING = _LazyConfigMapping(GENOMICS_MAPPING_NAMES)
SNP_MAPPING = _LazyConfigMapping(SNP_MAPPING_NAMES)
OTU_MAPPING = _LazyConfigMapping(OTU_MAPPING_NAMES)
MALDI_MAPPING = _LazyConfigMapping(MALDI_MAPPING_NAMES)
MS2_MAPPING = _LazyConfigMapping(MS2_MAPPING_NAMES)
PROTEOMICS_MAPPING = _LazyConfigMapping(PROTEOMICS_MAPPING_NAMES)
METAGENOMICS_MAPPING = _LazyConfigMapping(METAGENOMICS_MAPPING_NAMES)
MS1_MAPPING = _LazyConfigMapping(MS1_MAPPING_NAMES)
METABOLOMICS_MAPPING = _LazyConfigMapping(METABOLOMICS_MAPPING_NAMES)
RNA_SEQ_PLOTTER_MAPPING = _LazyConfigMapping(RNA_SEQ_PLOTTER_MAPPING_NAMES)
KMER_PLOTTER_MAPPING = _LazyConfigMapping(KMER_PLOTTER_MAPPING_NAMES)
BIODATA_PLOTTER_MAPPING = _LazyConfigMapping(BIODATA_PLOTTER_MAPPING_NAMES)
GENOMICS_PLOTTER_MAPPING = _LazyConfigMapping(GENOMICS_PLOTTER_MAPPING_NAMES)
SNP_PLOTTER_MAPPING = _LazyConfigMapping(SNP_PLOTTER_MAPPING_NAMES)
OTU_PLOTTER_MAPPING = _LazyConfigMapping(OTU_PLOTTER_MAPPING_NAMES)
MALDI_PLOTTER_MAPPING = _LazyConfigMapping(MALDI_PLOTTER_MAPPING_NAMES)
MS2_PLOTTER_MAPPING = _LazyConfigMapping(MS2_PLOTTER_MAPPING_NAMES)
PROTEOMICS_PLOTTER_MAPPING = _LazyConfigMapping(PROTEOMICS_PLOTTER_MAPPING_NAMES)
METAGENOMICS_PLOTTER_MAPPING = _LazyConfigMapping(METAGENOMICS_PLOTTER_MAPPING_NAMES)
MS1_PLOTTER_MAPPING = _LazyConfigMapping(MS1_PLOTTER_MAPPING_NAMES)
METABOLOMICS_PLOTTER_MAPPING = _LazyConfigMapping(METABOLOMICS_PLOTTER_MAPPING_NAMES)


EXPERIMENT_CONFIG_MAPPING = {
    "rna-seq": RNA_SEQ_MAPPING,
    "kmer": KMER_MAPPING,
    "biodata": BIODATA_MAPPING,
    "genomics": GENOMICS_MAPPING,
    "snp": SNP_MAPPING,
    "otu": OTU_MAPPING,
    "maldi": MALDI_MAPPING,
    "ms2": MS2_MAPPING,
    "proteomics": PROTEOMICS_MAPPING,
    "metagenomics": METAGENOMICS_MAPPING,
    "ms1": MS1_MAPPING,
    "metabolomics": METABOLOMICS_MAPPING,
}

EXPERIMENT_CONFIG_MAPPING_NAMES = {
    "rna-seq": RNA_SEQ_MAPPING_NAMES,
    "kmer": KMER_MAPPING_NAMES,
    "biodata": BIODATA_MAPPING_NAMES,
    "genomics": GENOMICS_MAPPING_NAMES,
    "snp": SNP_MAPPING_NAMES,
    "otu": OTU_MAPPING_NAMES,
    "maldi": MALDI_MAPPING_NAMES,
    "ms2": MS2_MAPPING_NAMES,
    "proteomics": PROTEOMICS_MAPPING_NAMES,
    "metagenomics": METAGENOMICS_MAPPING_NAMES,
    "ms1": MS1_MAPPING_NAMES,
    "metabolomics": METABOLOMICS_MAPPING_NAMES,
}

EXPERIMENT_PLOTTER_CONFIG_MAPPING = {
    "rna-seq": RNA_SEQ_PLOTTER_MAPPING,
    "kmer": KMER_PLOTTER_MAPPING,
    "biodata": BIODATA_PLOTTER_MAPPING,
    "genomics": GENOMICS_PLOTTER_MAPPING,
    "snp": SNP_PLOTTER_MAPPING,
    "otu": OTU_PLOTTER_MAPPING,
    "maldi": MALDI_PLOTTER_MAPPING,
    "ms2": MS2_PLOTTER_MAPPING,
    "proteomics": PROTEOMICS_PLOTTER_MAPPING,
    "metagenomics": METAGENOMICS_PLOTTER_MAPPING,
    "ms1": MS1_PLOTTER_MAPPING,
    "metabolomics": METABOLOMICS_PLOTTER_MAPPING,
}

EXPERIMENT_PLOTTER_CONFIG_MAPPING_NAMES = {
    "rna-seq": RNA_SEQ_PLOTTER_MAPPING_NAMES,
    "kmer": KMER_PLOTTER_MAPPING_NAMES,
    "biodata": BIODATA_PLOTTER_MAPPING_NAMES,
    "genomics": GENOMICS_PLOTTER_MAPPING_NAMES,
    "snp": SNP_PLOTTER_MAPPING_NAMES,
    "otu": OTU_PLOTTER_MAPPING_NAMES,
    "maldi": MALDI_PLOTTER_MAPPING_NAMES,
    "ms2": MS2_PLOTTER_MAPPING_NAMES,
    "proteomics": PROTEOMICS_PLOTTER_MAPPING_NAMES,
    "metagenomics": METAGENOMICS_PLOTTER_MAPPING_NAMES,
    "ms1": MS1_PLOTTER_MAPPING_NAMES,
    "metabolomics": METABOLOMICS_PLOTTER_MAPPING_NAMES,
}


EXPERIMENT_PREPROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        (
            "otu",
            [
                "row_abundance",
                "min_prevalence_feature_selector",
                "css",
            ],
        ),
        (
            "asv",
            [
                "row_abundance",
                "min_prevalence_feature_selector",
                "css",
            ],
        ),
        (
            "abundance",
            [
                "row_abundance",
                "min_prevalence_feature_selector",
                "css",
            ],
        ),
        (
            "metagenomics",
            [
                "row_abundance",
                "min_prevalence_feature_selector",
                "css",
            ],
        ),
        (
            "snp",
            [
                "min_prevalence_sample_filter",
                "min_prevalence_feature_selector",
            ],
        ),
        (
            "genomics",
            [
                "min_prevalence_sample_filter",
                "min_prevalence_feature_selector",
            ],
        ),
        (
            "maldi",
            [
                "min_prevalence_sample_filter",
                "relative_abundance",
                "min_prevalence_feature_selector",
            ],
        ),
        (
            "metabolomics",
            [
                "min_prevalence_sample_filter",
                "min_prevalence_feature_selector",
            ],
        ),
        (
            "proteomics",
            [
                "min_prevalence_sample_filter",
                "min_prevalence_feature_selector",
            ],
        ),
        (
            "transcriptomics",
            [
                "min_prevalence_sample_filter",
                "min_prevalence_feature_selector",
            ],
        ),
    ]
)


class _LazyLoadAllMappings(OrderedDict):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        if self._initialized:
            return
        warnings.warn(
            "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers. It does not contain all available model checkpoints, far from it. Checkout hf.co/models for that.",
            FutureWarning,
        )
        for processor_name, map_name in self._mapping.items():
            processor_category = PROCESSOR_CATEGORY_MAPPING_NAMES.get(
                processor_name, "models"
            )
            processor_type = PROCESSOR_TYPE_MAPPING_NAMES.get(processor_name, None)
            try:
                module_name = processor_name.replace("-", "_")
                package = (
                    f"biofit.{processor_category}"
                    if not processor_type
                    else f"biofit.{processor_category}.{processor_type}"
                )
                module = importlib.import_module(f".{module_name}", package)
                mapping = getattr(module, map_name)
            except ImportError:
                if is_transformers_available() and processor_category == "models":
                    from transformers.models.auto.configuration_auto import (
                        model_type_to_module_name,
                    )

                    module_name = model_type_to_module_name(processor_name)
                    module = importlib.import_module(
                        f".{module_name}", "transformers.models"
                    )
                    mapping = getattr(module, map_name)
            self._data.update(mapping)
        self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return self._data[key]

    def keys(self):
        self._initialize()
        return self._data.keys()

    def values(self):
        self._initialize()
        return self._data.values()

    def items(self):
        self._initialize()
        return self._data.keys()

    def __iter__(self):
        self._initialize()
        return iter(self._data)

    def __contains__(self, item):
        self._initialize()
        return item in self._data


def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and (not use_model_types):
        raise ValueError(
            "Using `use_model_types=False` requires a `config_to_class` dictionary."
        )
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {
                model_type: f"[`{config}`]"
                for model_type, config in CONFIG_MAPPING_NAMES.items()
            }
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in PROCESSOR_MAPPING_NAMES
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({PROCESSOR_MAPPING_NAMES[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: PROCESSOR_MAPPING_NAMES[model_type]
            for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        if docstrings is None:
            return fn
        lines = docstrings.split("\n")
        i = 0
        while (
            i < len(lines) and re.search("^(\\s*)List options\\s*$", lines[i]) is None
        ):
            i += 1
        if i < len(lines):
            indent = re.search("^(\\s*)List options\\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(
                indent, config_to_class=config_to_class, use_model_types=use_model_types
            )
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    """
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_processor(
        cls,
        model_type: str,
        *args,
        dataset_or_experiment: Union[str, "Bioset"] = None,
        **kwargs,
    ):
        if dataset_or_experiment is not None:
            experiment_name = get_experiment_name(dataset_or_experiment)
            if experiment_name in EXPERIMENT_CONFIG_MAPPING:
                mapper = EXPERIMENT_CONFIG_MAPPING[experiment_name]
                if model_type in mapper:
                    _config_class = mapper[model_type]
                    return _config_class(*args, **kwargs)
                raise ValueError(
                    f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(mapper.keys())}"
                )
        if model_type in CONFIG_MAPPING:
            _config_class = CONFIG_MAPPING[model_type]
            cls_kwargs = get_kwargs(kwargs, _config_class.__init__)
            return _config_class(*args, **cls_kwargs)
        elif (
            is_transformers_available()
            and PROCESSOR_CATEGORY_MAPPING_NAMES.get(model_type, "models") == "models"
        ):
            from transformers.models.auto.configuration_auto import (
                AutoConfig as HfAutoConfig,
            )

            return HfAutoConfig.for_model(model_type, *args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if is_transformers_available():
            from transformers.models.auto.configuration_auto import (
                AutoConfig as HfAutoConfig,
            )

            return HfAutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        raise EnvironmentError(
            "Using `AutoConfig.from_pretrained` requires the transformers library to be installed. You can install it with `pip install transformers`."
        )

    @staticmethod
    def register(processor_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        types = ProcessorConfig
        if is_transformers_available():
            from transformers import PretrainedConfig

            types = (PretrainedConfig, ProcessorConfig)
        config_processor_type = getattr(
            config, "processor_type", getattr(config, "model_type", None)
        )
        if issubclass(config, types) and config_processor_type != processor_type:
            raise ValueError(
                f"The config you are passing has a `model_type` attribute that is not consistent with the model type you passed (config has {config_processor_type} and you passed {processor_type}. Fix one of those so they match!"
            )
        CONFIG_MAPPING.register(processor_type, config, exist_ok=exist_ok)

    @staticmethod
    def register_experiment(experiment_name, processor_names, configs, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if not isinstance(configs, (list, tuple)):
            configs = [configs]
        if not isinstance(processor_names, (list, tuple)):
            processor_names = [processor_names]

        EXPERIMENT_CONFIG_MAPPING[experiment_name] = EXPERIMENT_CONFIG_MAPPING.get(
            experiment_name, _LazyConfigMapping(OrderedDict())
        )

        for processor_name, config in zip(processor_names, configs):
            if (
                processor_name in EXPERIMENT_CONFIG_MAPPING[experiment_name]
                and not exist_ok
            ):
                raise ValueError(
                    f"'{processor_name}' is already used by a processor config, pick another name."
                )
            EXPERIMENT_CONFIG_MAPPING[experiment_name].register(
                processor_name, config, exist_ok=exist_ok
            )


class AutoPlotterConfig:
    """
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoPlotterConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    @classmethod
    def for_experiment(cls, experiment: Union[str, "Bioset"], *args, **kwargs):
        experiment = get_experiment_name(experiment)
        if experiment in EXPERIMENT_PREPROCESSOR_MAPPING_NAMES:
            preprocessors = EXPERIMENT_PREPROCESSOR_MAPPING_NAMES[experiment]
            mapper = EXPERIMENT_PLOTTER_CONFIG_MAPPING[experiment]
            classes = []
            for preprocessor in preprocessors:
                _config_class = mapper[preprocessor]
                classes.append(_config_class(*args, **kwargs))
            return classes

        raise ValueError(
            f"Unrecognized dataset identifier: {experiment}. Should contain one of {', '.join(EXPERIMENT_PREPROCESSOR_MAPPING_NAMES.keys())}"
        )

    @classmethod
    def from_dataset(cls, dataset: "Bioset", *args, **kwargs):
        return cls.from_bioset(dataset, *args, **kwargs)

    @classmethod
    def from_bioset(cls, bioset: "Bioset", *args, **kwargs):
        if is_bioset(bioset):
            bioset = bioset._info.builder_name
        else:
            raise ValueError("Dataset should be a `biosets.Bioset` instance.")

        if bioset in EXPERIMENT_PREPROCESSOR_MAPPING_NAMES:
            preprocessors = EXPERIMENT_PREPROCESSOR_MAPPING_NAMES[bioset]
            mapper = EXPERIMENT_PLOTTER_CONFIG_MAPPING[bioset]
            classes = []
            for preprocessor in preprocessors:
                _config_class = mapper[preprocessor]
                classes.append(_config_class(*args, **kwargs))
            return classes

        raise ValueError(
            f"Unrecognized dataset identifier: {bioset}. Should contain one of {', '.join(EXPERIMENT_PREPROCESSOR_MAPPING_NAMES.keys())}"
        )

    @classmethod
    def for_processor(
        cls,
        processor_type: str,
        *args,
        dataset_or_experiment: Union[str, "Bioset"] = None,
        **kwargs,
    ):
        if dataset_or_experiment is not None:
            experiment_name = get_experiment_name(dataset_or_experiment)
            if experiment_name in EXPERIMENT_PLOTTER_CONFIG_MAPPING:
                mapper = EXPERIMENT_PLOTTER_CONFIG_MAPPING[experiment_name]
                if processor_type in mapper:
                    _config_class = mapper[processor_type]
                    return _config_class(*args, **kwargs)
                raise ValueError(
                    f"Unrecognized model identifier: {processor_type}. Should contain one of {', '.join(mapper.keys())}"
                )
        if processor_type in PLOTTER_CONFIG_MAPPING:
            _config_class = PLOTTER_CONFIG_MAPPING[processor_type]
            return _config_class(*args, **kwargs)
        elif (
            is_transformers_available()
            and PROCESSOR_CATEGORY_MAPPING_NAMES.get(processor_type, "models")
            == "models"
        ):
            from transformers.models.auto.configuration_auto import (
                AutoConfig as HfAutoConfig,
            )

            return HfAutoConfig.for_model(processor_type, *args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {processor_type}. Should contain one of {', '.join(PLOTTER_CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register_experiment(experiment_name, processor_names, configs, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if not isinstance(configs, (list, tuple)):
            configs = [configs]
        if not isinstance(processor_names, (list, tuple)):
            processor_names = [processor_names]

        EXPERIMENT_PLOTTER_CONFIG_MAPPING[experiment_name] = (
            EXPERIMENT_PLOTTER_CONFIG_MAPPING.get(
                experiment_name, _LazyConfigMapping(OrderedDict())
            )
        )

        for processor_name, config in zip(processor_names, configs):
            EXPERIMENT_PLOTTER_CONFIG_MAPPING[experiment_name].register(
                processor_name, config, exist_ok=exist_ok
            )


class AutoPreprocessorConfig(AutoConfig):
    """
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoPreprocessorConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    @classmethod
    def for_experiment(cls, experiment: Union[str, "Bioset"], *args, **kwargs):
        experiment = get_experiment_name(experiment)
        if experiment in EXPERIMENT_PREPROCESSOR_MAPPING_NAMES:
            preprocessors = EXPERIMENT_PREPROCESSOR_MAPPING_NAMES[experiment]
            mapper = EXPERIMENT_CONFIG_MAPPING[experiment]
            classes = []
            for preprocessor in preprocessors:
                _config_class = mapper[preprocessor]
                classes.append(_config_class(*args, **kwargs.get(preprocessor, {})))
            return classes

        raise ValueError(
            f"Unrecognized dataset identifier: {experiment}. Should contain one of {', '.join(EXPERIMENT_PREPROCESSOR_MAPPING_NAMES.keys())}"
        )

    @classmethod
    def register_pipeline(cls, experiment_name, preprocessor_names, exist_ok=False):
        """
        Register list of processors for an experiment.
        """
        if not isinstance(preprocessor_names, (list, tuple)):
            preprocessor_names = [preprocessor_names]

        if experiment_name in EXPERIMENT_PREPROCESSOR_MAPPING_NAMES and not exist_ok:
            raise ValueError(
                f"The pipeline for experiment {experiment_name} already exists. "
                "Use `exist_ok=True` to overwrite."
            )
        if experiment_name not in EXPERIMENT_CONFIG_MAPPING:
            raise ValueError(
                f"Experiment {experiment_name} does not have any configuration "
                "registered. Please register it via `AutoConfig.register_experiment`."
            )

        for preprocessor_name in preprocessor_names:
            if preprocessor_name not in EXPERIMENT_CONFIG_MAPPING[experiment_name]:
                raise ValueError(
                    f"Preprocessor {preprocessor_name} is not registered for "
                    f"experiment {experiment_name}. Please register it via "
                    "`AutoConfig.register_experiment`."
                )
        EXPERIMENT_PREPROCESSOR_MAPPING_NAMES[experiment_name] = preprocessor_names
