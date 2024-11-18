from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

import pyarrow as pa

from biofit.integration.biosets import get_feature
from biofit.integration.R import RCaller
from biofit.integration.R.r_caller import PackageNotInstalledError
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..scaling import Scaler, ScalerConfig

logger = logging.get_logger(__name__)


@dataclass
class TMMScalerConfig(ScalerConfig):
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    # process descriptions
    _fit_process_desc: str = field(
        default="Determining reference samples for TMM normalization",
        init=False,
        repr=False,
    )
    _transform_process_desc: str = field(
        default="Applying TMM normalization", init=False, repr=False
    )
    processor_name: str = field(default="tmm", init=False, repr=False)

    # attributes
    r_source: str = field(
        default=Path(__file__).with_suffix(".R").as_posix(), init=False, repr=False
    )
    fit_func_name: str = field(default="edger_tmm_fit", init=False, repr=False)
    cpm_transform_func_name: str = field(
        default="edger_tmm_cpm_transform", init=False, repr=False
    )
    tpm_transform_func_name: str = field(
        default="edger_tmm_tpm_transform", init=False, repr=False
    )

    log: bool = True
    prior_count: int = 2
    meta_col: str = "Length"
    gene_col: str = "genes"

    # estimated attributes
    ref_samples: Optional[pa.Table] = field(default=None, init=False, repr=False)


class TMMScalerConfigForMetagenomics(TMMScalerConfig):
    # dataset specific attributes
    _input_feature_types: List[Type] = (
        get_feature("Abundance"),
        get_feature("ReadCount"),
    )
    dataset_name = "metagenomics"


class TMMScalerConfigForOTU(TMMScalerConfig):
    # dataset specific attributes
    _input_feature_types: List[Type] = get_feature("Abundance")
    dataset_name = "otu"


class TMMScalerConfigForSNP(TMMScalerConfig):
    # dataset specific attributes
    _input_feature_types: List[Type] = get_feature("GenomicVariant")
    dataset_name = "snp"


class TMMScaler(Scaler):
    output_dtype = "float64"

    # config class
    config_class = TMMScalerConfig
    config: TMMScalerConfig

    def __init__(
        self,
        log: bool = True,
        prior_count: int = 2,
        meta_col: str = "Length",
        gene_col: str = "genes",
        config: Optional[TMMScalerConfig] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            log=log,
            prior_count=prior_count,
            meta_col=meta_col,
            gene_col=gene_col,
            **kwargs,
        )
        r_caller = RCaller.from_script(self.config.r_source)
        install_missing = kwargs.get("install_missing")
        try:
            r_caller.verify_r_dependencies(
                cran_dependencies=["BiocManager"],
                bioconductor_dependencies=["edgeR"],
                install_missing=install_missing,
            )
        except PackageNotInstalledError:
            raise PackageNotInstalledError(
                "TMMScale requires the following R package: edgeR. To "
                "install, initialize the TMMScaler with install_missing=True or run "
                "R -e 'BiocManager::install(\"edgeR\")' in your terminal."
            )
        self.fit_func = r_caller.get_method(self.config.fit_func_name)
        self.cpm_transform = r_caller.get_method(self.config.cpm_transform_func_name)
        self.tpm_transform = r_caller.get_method(self.config.tpm_transform_func_name)

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
    ):
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
        genes=None,
        input_columns: SelectedColumnTypes = None,
        gene_col: SelectedColumnTypes = None,
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
        self._input_columns = self._set_input_columns_and_arity(input_columns, gene_col)
        return self._process_transform(
            X,
            genes,
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
        genes=None,
        input_columns: SelectedColumnTypes = None,
        gene_col: SelectedColumnTypes = None,
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
            genes=genes,
            gene_col=gene_col,
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

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)

        # load file from src/biofit/preprocessing/scaling/tmm/tmm.R
        r_caller = RCaller.from_script(self.config.r_source)
        self.fit_func = r_caller.get_method(self.config.fit_func_name)
        self.cpm_transform = r_caller.get_method(self.config.cpm_transform_func_name)
        self.tpm_transform = r_caller.get_method(self.config.tpm_transform_func_name)
        return self

    def _fit_arrow(self, X: pa.Table):
        self.config.ref_samples = self.fit_func(X)
        return self

    def _transform_arrow(self, X: pa.Table, genes=None):
        if genes:
            return self.tpm_transform(
                X=X,
                feature_meta=genes,
                ref_samples=self.config.ref_samples,
                log=self.config.log,
                prior_count=self.config.prior_count,
                meta_col=self.config.meta_col,
            )
        else:
            return self.cpm_transform(
                X=X,
                ref_samples=self.config.ref_samples,
                log=self.config.log,
                prior_count=self.config.prior_count,
            )
