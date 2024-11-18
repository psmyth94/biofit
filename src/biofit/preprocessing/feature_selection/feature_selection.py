from dataclasses import dataclass, field

from biocore import DataHandler
from biocore.utils import get_kwargs

from biofit.processing import BaseProcessor, ProcessorConfig
from biofit.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class FeatureSelectorConfig(ProcessorConfig):
    processor_type: str = field(default="feature_selection", init=False, repr=False)


class FeatureSelector(BaseProcessor):
    """Base class for feature selection processors."""

    config_class = FeatureSelectorConfig
    config: FeatureSelectorConfig

    def run(self, X, runner=None, fn_kwargs: dict = {}, **map_kwargs):
        fn_kwargs = self._prepare_runner(X, **fn_kwargs)
        if "func_type" in fn_kwargs and fn_kwargs["func_type"] != "_fit":
            input_format = fn_kwargs["in_format_kwargs"]["target_format"]
            fn_kwargs["out_format_kwargs"]["target_format"] = input_format
            runner = None
        return super().run(X, runner=runner, fn_kwargs=fn_kwargs, **map_kwargs)

    def _transform_any(self, X, selected_indices=None):
        if selected_indices is None:
            return X
        return DataHandler.select_columns(X, selected_indices)

    def _process_transform_batch_input(self, X, *fn_args, **fn_kwargs):
        func = fn_kwargs.get("fn", None)
        in_format_kwargs = fn_kwargs.get("in_format_kwargs", {})
        in_format_kwargs["input_columns"] = None
        input = DataHandler.to_format(X, **in_format_kwargs)
        _fn_kwargs = get_kwargs(fn_kwargs, func)

        selected_indices = fn_kwargs.get("selected_indices", None)
        unused_indices = fn_kwargs.get("unused_indices", None)
        keep_unused_columns = fn_kwargs.get("keep_unused_columns", None)
        if self.config._feature_names_out and DataHandler.supports_named_columns(input):
            feature_idx_out = DataHandler.get_column_indices(
                input, self.config._feature_names_out, raise_if_missing=False
            )
        elif self.config._feature_idx_out is not None:
            feature_idx_out = [
                selected_indices[i] for i in self.config._feature_idx_out
            ]
        else:
            raise ValueError(
                "FeatureSelectorConfig requires either _feature_idx_out, as well as "
                "_feature_idx_out when input format supports named columns."
            )
        if keep_unused_columns:
            feature_idx_out = list(sorted(feature_idx_out + unused_indices))
        _fn_kwargs["selected_indices"] = feature_idx_out
        return input, fn_args, _fn_kwargs

    def _process_fit_output(self, input, out):
        idx_out = self.config._feature_idx_out
        names_in = self.config.feature_names_in_
        names_out = self.config._feature_names_out
        if idx_out is not None:
            if names_in is not None and (
                names_out is None or len(idx_out) < len(names_out)
            ):
                names_out = [names_in[i] for i in idx_out]
        self.config._feature_names_out = names_out

        return super()._process_fit_output(input, out)

    def _process_transform_batch_output(self, input, out, **fn_kwargs):
        # do nothing
        return out

    def _process_transform_output(self, output, input, *args, **kwargs):
        unused_indices = kwargs.get("unused_indices", None)
        new_fingerprint = kwargs.get("fingerprint", None)
        selected_indices = self.config._feature_idx_out
        if DataHandler.supports_named_columns(input) and self.config._feature_names_out:
            input_cols = DataHandler.get_column_names(input)
            col_dict = dict(zip(input_cols, range(len(input_cols))))
            selected_indices = {
                col_dict[name]
                for name in self.config._feature_names_out
                if name in col_dict
            }

        before_filtering = len(kwargs.get("selected_indices", []))
        logger.info(
            f'"{self.__class__.__name__}": Selected {len(selected_indices)} out of '
            f"{before_filtering} input features"
        )

        if unused_indices:
            selected_indices = set(unused_indices).union(selected_indices)

        return DataHandler.select_columns(
            input, sorted(list(selected_indices)), new_fingerprint=new_fingerprint
        )
