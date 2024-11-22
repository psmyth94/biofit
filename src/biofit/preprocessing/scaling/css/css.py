from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
from scipy.interpolate import interp1d

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..scaling import Scaler, ScalerConfig

logger = logging.get_logger(__name__)


def css_percentile_fast(mat, relative_threshold=0.1):
    """Calculates the percentile for which to sum counts up to and scale by.

    Args:

    """
    # Check for sample with one or zero features
    if np.any(np.sum(mat > 0, axis=1) <= 1):
        raise ValueError("Warning: sample with one or zero features")

    mat = mat.astype(float)
    # Sort each row with elements greater than zero
    smat = np.sort(mat * (mat > 0), axis=1)[:, ::-1]
    smat[smat == 0] = np.nan

    # Pad with NAs to make all rows the same length
    max_length = np.max(np.nansum(smat > 0, axis=1))
    padded_smat = np.full((smat.shape[0], max_length), np.nan)
    for i in range(smat.shape[0]):
        valid_values = smat[i, smat[i] > 0]
        padded_smat[i, : len(valid_values)] = valid_values

    # Calculate quantiles for each row
    quantiles = np.nanquantile(
        padded_smat, np.linspace(0, 1, padded_smat.shape[1]), axis=1
    )

    padded_smat[np.isnan(padded_smat)] = 0

    # Compute column means, ignoring NaNs
    ref1 = np.nanmean(padded_smat, axis=0)[::-1]

    # Calculate differences
    diffr = ref1[:, np.newaxis] - quantiles

    # Calculate median of absolute differences
    diffr1 = np.nanmedian(np.abs(diffr), axis=1)

    diffr1[diffr1 == 0] = np.nan

    # Determine threshold
    rel_diff = np.abs(np.diff(diffr1)) / diffr1[:-1]

    x = (np.where(rel_diff > relative_threshold)[0][0] + 1) / len(diffr1)
    if x <= 0.5:
        logger.info(
            f"Percentile calculated at {x}, which is less than 0.5. Using default value instead."
        )
        x = 0.5
    return x


def css_percentile(mat: np.ndarray, approx=False, relative_threshold=0.1):
    if np.any(np.sum(mat, axis=1) == 0):
        raise ValueError("Warning: empty feature", mat)

    # Sorting each row
    smat = np.sort(mat, axis=1)
    ref = np.mean(smat, axis=0)
    orig_dtype = mat.dtype
    mat = mat.astype(float) if not np.issubdtype(orig_dtype, np.floating) else mat

    if not mat.flags["WRITEABLE"]:
        # copy the array if it is not writeable
        mat = np.array(mat)
    mat[mat == 0] = np.nan

    refS = np.sort(ref)

    k = np.where(refS > 0)[0][0]
    lo = len(refS) - k

    if not approx:
        diffr = np.apply_along_axis(
            lambda row: refS[k:] - np.nanquantile(row, np.linspace(0, 1, lo), axis=0),
            axis=1,
            arr=mat,
        )
    else:

        def f(row):
            srow = np.sort(row)
            rrow = (srow[0], np.nanmax(srow))
            y = np.arange(len(row))
            return refS[k:] - interp1d(row, y)(np.linspace(*rrow, lo))

        diffr = np.apply_along_axis(f, axis=1, arr=mat)

    mat[np.isnan(mat)] = 0
    mat = mat.astype(orig_dtype) if not np.issubdtype(orig_dtype, np.floating) else mat

    diffr2 = np.nanmedian(np.abs(diffr), axis=0)

    rel_diff = np.abs(np.diff(diffr2)) / diffr2[:-1]
    if len(rel_diff) == 0:
        return 0.5
    x = (np.where(rel_diff > relative_threshold)[0][0] + 1) / len(diffr2)
    if x <= 0.5:
        logger.info(
            f"Percentile calculated at {x}, which is less than 0.5. Using default value instead."
        )
        x = 0.5

    return x


@dataclass
class CumulativeSumScalerConfig(ScalerConfig):
    """
    Configuration for CumulativeSumScaler.
    """

    processor_name: str = field(default="css", init=False, repr=False)
    _fit_process_desc: str = field(
        default="Calculating cumulative sum scaling percentile", init=False, repr=False
    )
    _transform_process_desc: str = field(
        default="Applying cumulative sum scaling", init=False, repr=False
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )

    scale: int = 1000
    relative_threshold: float = 0.5
    approx: bool = False
    percentile: float = None


@dataclass
class CumulativeSumScalerConfigForMetagenomics(CumulativeSumScalerConfig):
    """
    CumulativeSumScaler specifically designed for metagenomics data.
    """

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [(get_feature("ReadCount"), get_feature("Abundance"))],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [(get_feature("ReadCount"), get_feature("Abundance"))],
        init=False,
        repr=False,
    )
    experiment_name: str = field(default="metagenomics", init=False, repr=False)


@dataclass
class CumulativeSumScalerConfigForOTU(CumulativeSumScalerConfig):
    """
    CumulativeSumScaler specifically designed for otu abundance.

    Args:
    """

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    experiment_name: str = field(default="otu", init=False, repr=False)


class CumulativeSumScaler(Scaler):
    """
    BaseCumSumScaler applies cumulative sum scaling to the input data.
    """

    output_dtype = "float64"
    _config_class = CumulativeSumScalerConfig
    config: CumulativeSumScalerConfig

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        if self.config.percentile is not None:
            # there is overhead when passing datasets to the fit method
            # (e.g. converting one format to another, preparing mapping jobs, etc.)
            # which is unnecessary if the percentile is already known
            # Therefore, we remove the fit method to avoid it being called
            # TODO: This is a temporary solution. We should find a better way to handle this
            delattr(self, "fit_numpy")
        return self

    def fit(
        self,
        X,
        input_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        num_proc: int = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ) -> "CumulativeSumScaler":
        self.config._input_columns = self._set_input_columns_and_arity(input_columns)
        return self._process_fit(
            X,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            num_proc=num_proc,
            map_kwargs=map_kwargs,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            fingerprint=fingerprint,
        )

    def transform(
        self,
        X,
        input_columns: SelectedColumnTypes = None,
        keep_unused_columns: bool = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        cache_dir: str = None,
        cache_file_name: str = None,
        load_from_cache_file: bool = None,
        batched: bool = None,
        batch_size: int = None,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = None,
        num_proc: int = None,
        fingerprint: str = None,
    ):
        self._input_columns = self._set_input_columns_and_arity(input_columns)
        return self._process_transform(
            X,
            keep_unused_columns=keep_unused_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            output_format=output_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    def fit_transform(
        self,
        X,
        *args,
        input_columns: SelectedColumnTypes = None,
        keep_unused_columns: bool = True,
        raise_if_missing: bool = False,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        num_proc: int = None,
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ):
        return self.fit(
            X,
            input_columns=input_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            num_proc=num_proc,
            map_kwargs=map_kwargs,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            fingerprint=fingerprint,
        ).transform(
            X,
            input_columns=input_columns,
            keep_unused_columns=keep_unused_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            output_format=output_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    def _fit_numpy(self, X: np.ndarray):
        # The fast implementation of CSS requires counts of all samples to at least
        # have two non zero features.
        if np.any(np.sum(X > 0, axis=1) <= 1):
            self.config.percentile = css_percentile(X)
        else:
            self.config.percentile = css_percentile_fast(X)
        return self

    def _transform_numpy(self, X: np.ndarray):
        xx = np.where(X == 0, np.nan, X)

        # Compute row quantiles
        qs = np.nanquantile(xx, self.config.percentile, axis=1, keepdims=True)

        xx_adj = xx - np.finfo(float).eps
        norm_factors = np.nansum(np.where(xx_adj <= qs, xx_adj, 0), axis=1)
        norm_factors[norm_factors == 0] = np.nan
        norm_factors = norm_factors / self.config.scale

        return X / norm_factors[:, np.newaxis]
