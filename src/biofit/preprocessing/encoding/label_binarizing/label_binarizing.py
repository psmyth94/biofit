from dataclasses import dataclass, field
from typing import List, Type, Union

from biocore.utils.import_util import is_biosets_available
import numpy as np
import pyarrow as pa
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..encoding import Encoder, EncoderConfig

logger = logging.get_logger(__name__)


@dataclass
class LabelBinarizerConfig(EncoderConfig):
    """Configuration class for label binarizer."""

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ClassLabel")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ClassLabel")], init=False, repr=False
    )

    processor_name: str = field(default="label_binarizing", init=False, repr=False)
    _transform_process_desc: str = field(default=None, init=False, repr=False)
    positive_labels: list = field(default_factory=list)
    negative_labels: list = field(default_factory=list)
    as_one_hot: bool = False
    names: list = field(default_factory=list, init=False, repr=False)

    # auto attributes
    label_mapping: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.as_one_hot:
            self._transform_process_desc = "One-hot encoding labels"
            self._fit_process_desc = "Determining unique labels"
            if self.names:
                self._transform_process_desc = (
                    f"One-hot encoding labels with {self.names}"
                )
                self.label_mapping = {
                    str(label): i for i, label in enumerate(self.names)
                }
        else:
            if not isinstance(self.positive_labels, list):
                self.positive_labels = [self.positive_labels]
            if not isinstance(self.negative_labels, list):
                self.negative_labels = [self.negative_labels]
            if len(self.positive_labels) < 1 and len(self.negative_labels) < 1:
                raise ValueError(
                    "At least one of positive_labels or negative_labels must be provided"
                )

            self._transform_process_desc = f"Binarizing labels with positive_labels={self.positive_labels} and negative_labels={self.negative_labels}"

            if self.positive_labels:
                self._transform_process_desc = (
                    f"Binarizing labels with positive_labels={self.positive_labels}"
                )

                if self.negative_labels:
                    self._transform_process_desc += (
                        f" and negative_labels={self.negative_labels}"
                    )
            else:
                self._transform_process_desc = (
                    f"Binarizing labels with negative_labels={self.negative_labels}"
                )

            if not isinstance(self.positive_labels, list):
                self.positive_labels = [self.positive_labels]

            if not isinstance(self.negative_labels, list):
                self.negative_labels = [self.negative_labels]

            self.label_mapping = {str(label): 1 for label in self.positive_labels}
            self.label_mapping.update({str(label): 0 for label in self.negative_labels})


class LabelBinarizer(Encoder):
    _config_class = LabelBinarizerConfig
    config: LabelBinarizerConfig

    def __init__(
        self,
        positive_labels: list = [],
        negative_labels: list = [],
        names: list = [],
        as_one_hot: bool = False,
        config: LabelBinarizerConfig = None,
        **kwargs,
    ):
        """
        Args:
            positive_labels (list, *optional*):
                The labels to be considered as positive. Default is `[]`.
            negative_labels (list, *optional*):
                The labels to be considered as negative. Default is `[]`.
            as_one_hot (bool, *optional*):
                Whether to encode the labels as one-hot vectors. Default is `False`.
            names (list, *optional*):
                The names of the classes. This is required when `as_one_hot` is `True`.
            config (LabelBinarizerConfig, *optional*):
            **kwargs:
                Arguments that are passed to ProcessorConfig.
        """
        super().__init__(
            config=config,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            as_one_hot=as_one_hot,
            names=names,
            **kwargs,
        )
        self.output_feature_type = get_feature("BinClassLabel")

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.config.__post_init__()
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
    ) -> "LabelBinarizer":
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
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            fingerprint=fingerprint,
        ).transform(
            X,
            input_columns=input_columns,
            keep_unused_columns=keep_unused_columns,
            raise_if_missing=raise_if_missing,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            output_format=output_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    def _get_features_out(self, X, **kwargs):
        kwargs.pop("one_to_one_features", None)
        features = super()._get_features_out(X, one_to_one_features=True, **kwargs)

        input_names = DataHandler.get_column_names(X, generate_cols=True)
        input_names = [input_names[i] for i in self._selected_indices]
        if self.config.as_one_hot and self.config.names:
            # incase that the user provided the pre-encoded labels of get_feature("ClassLabel")
            for name in input_names:
                label_feature = features.pop(name, None)
                if isinstance(label_feature, get_feature("ClassLabel")):
                    label_names = label_feature.names
                    if self.config.names and isinstance(self.config.names[0], str):
                        self.config.label_mapping = {
                            str(label_names.index(label)): i
                            for i, label in enumerate(self.config.names)
                        }
            features.update(
                {
                    name: get_feature("ClassLabel")(names=["negative", "positive"])
                    for name in self.config.names
                }
            )
            return features

        if is_biosets_available():
            for name in input_names:
                # incase that the user provided the pre-encoded labels of get_feature("ClassLabel")
                label_feature = features.get(name)
                if isinstance(label_feature, get_feature("ClassLabel")):
                    label_names = label_feature.names
                    if self.config.positive_labels and isinstance(
                        self.config.positive_labels[0], str
                    ):
                        self.config.label_mapping = {
                            str(label_names.index(label)): 1
                            for label in self.config.positive_labels
                        }
                    if self.config.negative_labels and isinstance(
                        self.config.negative_labels[0], str
                    ):
                        self.config.label_mapping.update(
                            {
                                str(label_names.index(label)): 0
                                for label in self.config.negative_labels
                            }
                        )
                label_feature = get_feature("BinClassLabel")(
                    names=self.config.names or ["negative", "positive"],
                    positive_labels=self.config.positive_labels,
                    negative_labels=self.config.negative_labels,
                )
                features[name] = label_feature

        return features

    def _process_fit_input(self, input, **kwargs):
        if not self.config.as_one_hot or self.config.names:
            kwargs["fn_kwargs"]["fn"] = None
        if self.config.names:
            self.config.label_mapping = {
                str(label): i for i, label in enumerate(self.config.names)
            }
        return super()._process_fit_input(input, **kwargs)

    def _check_data(self, X):
        X_dims = DataHandler.get_shape(X)
        if len(X_dims) > 1:
            if X_dims[1] == 1:
                out = DataHandler.select_column(X, 0)
            else:
                raise ValueError(
                    f"Expected input to have 1 column, got {X_dims[1]} columns"
                )
        else:
            out = X
        return DataHandler.to_format(out, "list")

    def _fit_array(self, X):
        out = self._check_data(X)
        self.config.names = DataHandler.to_format(DataHandler.unique(out), "list")
        self.config._transform_process_desc = (
            f"Encoding labels with: {self.config.label_mapping}"
        )
        return self

    def _partial_fit_array(self, X):
        out = self._check_data(X)
        labs = DataHandler.to_format(DataHandler.unique(out), "list")
        self.config.names = np.unique(self.config.names + labs).tolist()
        return self

    def _pool_fit(self, fitted_processors):
        names = []
        for processor in fitted_processors:
            if isinstance(processor, LabelBinarizer):
                names += processor.config.names
        self.config.names = np.unique(names).tolist()
        self.config.label_mapping = {
            str(label): i for i, label in enumerate(self.config.names)
        }
        self.config._transform_process_desc = (
            f"Encoding labels with: {self.config.label_mapping}"
        )
        return self

    def _process_fit_output(self, input, out):
        if self.config.as_one_hot:
            self.config._n_features_out = len(self.config.names)
            self.output_dtype = "int64"
        else:
            self.config._n_features_out = None
        self.config.label_mapping = {
            str(k): v for k, v in self.config.label_mapping.items()
        }
        return super()._process_fit_output(input, out)

    def _one_hot_transform(self, X):
        col = self._check_data(X)

        labs = np.zeros((len(col), len(self.config.names) + 1), dtype=np.int64)

        inds = [self.config.label_mapping.get(str(val), -1) for val in col]
        labs[np.arange(len(col)), inds] = 1
        return labs[:, :-1]

    def _binarize_transform(self, X: Union[pa.Table, pa.Array]):
        col = self._check_data(X)

        if self.config.positive_labels and self.config.negative_labels:
            labs = [-1] * len(col)
        elif self.config.positive_labels:
            labs = [0] * len(col)
        else:
            labs = [1] * len(col)

        if self.config.positive_labels and self.config.negative_labels:
            labs = [
                self.config.label_mapping.get(str(val), -1)
                if labs[i] == -1
                else labs[i]
                for i, val in enumerate(col)
            ]
        elif self.config.positive_labels:
            labs = [
                self.config.label_mapping.get(str(val), 0) if labs[i] == 0 else labs[i]
                for i, val in enumerate(col)
            ]
        else:
            labs = [
                self.config.label_mapping.get(str(val), 1) if labs[i] == 1 else labs[i]
                for i, val in enumerate(col)
            ]

        return pa.array(labs)

    def _transform_array(self, X):
        if not self.config.as_one_hot:
            return self._binarize_transform(X)
        else:
            return self._one_hot_transform(X)
