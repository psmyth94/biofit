# Portions of this module are derived from the Pyckmeans project available at:
# https://github.com/TankredO/pyckmeans

# Pyckmeans is licensed under the MIT License. The following is a copy of the original license under which the Pyckmeans software is distributed:

# MIT License

# Copyright (c) 2021 Tankred Ott

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Principal Coordinate Analysis (PCoA) feature extraction module."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.stat import DistanceStat, DistanceStatConfig
from biofit.utils import logging

from ..feature_extraction import FeatureExtractor, FeatureExtractorConfig

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


def _center_mat(dmat: np.ndarray) -> np.ndarray:
    """_center_mat

    Center n*n matrix.

    Parameters
    ----------
    dmat : np.ndarray
        n*n matrix.

    Returns
    -------
    np.ndarray
        Centered matrix.
    """

    n = dmat.shape[0]
    mat = np.full((n, n), -1 / n)
    mat[np.diag_indices(n)] += 1

    return mat.dot(dmat).dot(mat)


class InvalidCorrectionTypeError(Exception):
    """InvalidCorrectionTypeError"""


class NegativeEigenvaluesCorrectionError(Exception):
    """FailedCorrectionError

    Error, signalling that the correction of negative eigenvalues failed.
    """


class NegativeEigenvaluesWarning(Warning):
    """NegativeEigenvaluesWarning

    Warning, signalling that negative eigenvalues were encountered.
    """


def pcoa(
    x: np.ndarray,
    correction: Optional[str] = None,
    eps: float = 1e-8,
):
    """pcoa

    Principle Coordinate Analysis.

    Parameters
    ----------
    dist : Union[np.ndarray, pyckmeans.distance.DistanceMatrix]
        n*n distance matrix either as np ndarray or as pyckmeans DistanceMatrix.
    correction: Optional[str]
        Correction for negative eigenvalues, by default None.
        Available corrections are:
            - None: negative eigenvalues are set to 0
            - lingoes: Lingoes correction
            - cailliez: Cailliet correction
    eps : float, optional
        Eigenvalues smaller than eps will be dropped. By default 0.0001

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors.

    Raises
    ------
    InvalidCorrectionTypeError
        Raised if an unknown correction type is passed.
    NegativeEigenvaluesCorrectionError
        Raised if correction parameter is set and correction of negative
        eigenvalues is not successful.
    """

    if correction is not None and correction not in ["lingoes", "cailliez"]:
        msg = (
            f'Unknown correction type "{correction}". '
            + 'Available correction types are: "lingoes", "cailliez"'
        )
        raise InvalidCorrectionTypeError(msg)

    # center matrix
    dmat_centered = _center_mat((x * x) / -2)

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(dmat_centered, "U")

    # order descending
    ord_idcs = np.argsort(eigvals)[::-1]
    eigvals = eigvals[ord_idcs]
    eigvecs = eigvecs[:, ord_idcs]

    # get min eigenvalue
    min_eigval = np.min(eigvals)

    # set small eigenvalues to 0
    zero_eigval_idcs = np.nonzero(np.abs(eigvals) < eps)[0]
    eigvals[zero_eigval_idcs] = 0

    # no negative eigenvalues
    if min_eigval > -eps:
        fze_idx = len(np.nonzero(eigvals > eps)[0])  # index of first zero in eigvals
        vectors = eigvecs[:, :fze_idx] * np.sqrt(eigvals[:fze_idx])

        return eigvals, vectors

    # negative eigenvalues
    else:
        fze_idx = len(np.nonzero(eigvals > eps)[0])  # index of first zero in eigvals
        vectors = eigvecs[:, :fze_idx] * np.sqrt(eigvals[:fze_idx])

        # negative eigenvalues, no correction
        if not correction:
            logger.warn(
                "Negative eigenvalues encountered but no correction applied. "
                "Negative eigenvalues will be treated as 0."
            )

            return eigvals, vectors

        # negative eigenvalues, correction

        # -- correct distance matrix
        # lingoes correction
        if correction == "lingoes":
            corr_1 = -min_eigval

            # corrected distance matrix
            x_ncorr = -0.5 * ((x * x) + 2 * corr_1)
        elif correction == "cailliez":
            dmat_centered_2 = _center_mat(-0.5 * x)

            # prepare matrix for correction
            upper = np.c_[np.zeros((x.shape[0], x.shape[0])), 2 * dmat_centered]
            lower = np.c_[np.diag(np.full(x.shape[0], -1)), -4 * dmat_centered_2]
            sp_mat = np.r_[upper, lower]

            corr_2 = np.max(np.real(np.linalg.eigvals(sp_mat)))

            # corrected distance matrix
            x_ncorr = -0.5 * (x + corr_2) ** 2

        # -- apply PCoA to corrected distance matrix
        x_ncorr[np.diag_indices(x_ncorr.shape[0])] = 0
        x_ncorr = _center_mat(x_ncorr)

        eigvals_ncorr, eigvecs_ncorr = np.linalg.eigh(x_ncorr, "U")

        # order descending
        ord_idcs_ncorr = np.argsort(eigvals_ncorr)[::-1]
        eigvals_ncorr = eigvals_ncorr[ord_idcs_ncorr]
        eigvecs_ncorr = eigvecs_ncorr[:, ord_idcs_ncorr]

        # get min eigenvalue
        min_eigval_ncorr = np.min(eigvals_ncorr)

        # set small eigenvalues to 0
        zero_eigval_idcs_ncorr = np.nonzero(np.abs(eigvals_ncorr) < eps)[0]
        eigvals_ncorr[zero_eigval_idcs_ncorr] = 0

        if min_eigval_ncorr < -eps:
            msg = (
                "Correction failed. There are still negative eigenvalues after applying "
                + f"{correction.capitalize()} correction."
            )
            raise NegativeEigenvaluesCorrectionError(msg)

        fze_idx_ncorr = len(
            np.nonzero(eigvals_ncorr > eps)[0]
        )  # index of first zero in eigvals
        vectors_ncorr = eigvecs_ncorr[:, :fze_idx_ncorr] * np.sqrt(
            eigvals_ncorr[:fze_idx_ncorr]
        )

        return eigvals_ncorr, vectors_ncorr


@dataclass
class PCoAFeatureExtractorConfig(FeatureExtractorConfig):
    processor_name: str = field(default="pcoa", init=False, repr=False)
    output_template_name: str = field(default="Dim{i+1}", init=False, repr=False)
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
    _fit_process_desc: str = field(
        default="",
        init=False,
        repr=False,
    )
    _transform_process_desc: str = field(
        default="Applying Principal Coordinate Analysis (PCoA) to the input data",
        init=False,
        repr=False,
    )
    n_components: int = None
    correction: str = None
    eps: float = 1e-3

    metric: str = "braycurtis"
    p: float = 2
    w: float = None
    V: float = None
    VI: float = None
    squareform: bool = True

    # fitted attributes
    vectors: np.ndarray = None
    eigvals: np.ndarray = None

    def __post_init__(self):
        self._fit_process_desc = f"Calculating PCoA using {self.metric} distance"


@dataclass
class PCoAFeatureExtractorConfigForOTU(PCoAFeatureExtractorConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    dataset_name: str = field(default="otu", init=False, repr=False)
    correction: str = "cailliez"


class PCoAFeatureExtractor(FeatureExtractor):
    output_dtype = "float64"

    config_class = PCoAFeatureExtractorConfig
    config: PCoAFeatureExtractorConfig

    def __init__(
        self,
        n_components: int = None,
        correction: str = None,
        eps: float = 1e-3,
        metric: str = "braycurtis",
        p: float = 2,
        w: float = None,
        V: float = None,
        VI: float = None,
        squareform: bool = True,
        config: Optional[PCoAFeatureExtractorConfig] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            n_components=n_components,
            correction=correction,
            eps=eps,
            metric=metric,
            p=p,
            w=w,
            V=V,
            VI=VI,
            squareform=squareform,
            **kwargs,
        )
        distance_config = DistanceStatConfig.from_config(self.config)
        self.distance = DistanceStat(distance_config)
        self.config._n_features_out = self.config.n_components

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        distance_config = DistanceStatConfig.from_config(self.config)
        self.distance = DistanceStat(distance_config)
        self.config._n_features_out = self.config.n_components
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
    ) -> "PCoAFeatureExtractor":
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

    def _fit_numpy(self, X: "np.ndarray"):
        self.config.eigvals, self.config.vectors = pcoa(
            self.distance._transform_numpy(X),
            correction=self.config.correction,
            eps=self.config.eps,
        )

        return self

    def _process_fit_output(self, input, output):
        if self.config.n_components is None:
            self.config.n_components = self.config.vectors.shape[1]
        self.config._n_features_out = self.config.n_components
        return super()._process_fit_output(input, output)

    def _transform_numpy(self, X: "np.ndarray"):
        return self.config.vectors[:, : self.config.n_components]
