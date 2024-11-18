from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
from biocore import DataHandler
from biocore.utils.import_util import is_datasets_available
from biocore.utils.inspect import get_kwargs
from biocore.utils.py_util import is_bioset, is_dataset, is_iterable_dataset

from biofit.processing import BaseProcessor, ProcessorConfig
from biofit.utils import logging
from biofit.utils.table_util import string_to_arrow

logger = logging.get_logger(__name__)


@dataclass
class SampleFilterConfig(ProcessorConfig):
    processor_type: str = field(default="filtering", init=False, repr=False)


class SampleFilter(BaseProcessor):
    """Base class for filtering processors.

    NOTE: All transformation functions must return bools.
    """

    _feature_dependent = (
        False  # SampleFiltering does not depend on features when transforming
    )

    def run(self, X, runner=None, fn_kwargs: dict = {}, **map_kwargs):
        fn_kwargs = self._prepare_runner(X, **fn_kwargs)
        if fn_kwargs["func_type"] != "_fit":
            if is_datasets_available():
                if (
                    is_bioset(X) or is_dataset(X, iterable=False)
                ) and "out_type" not in fn_kwargs:
                    from datasets import Dataset

                    runner = Dataset.map
                    map_kwargs = get_kwargs(map_kwargs, runner)
                elif is_iterable_dataset(X):
                    from datasets import IterableDataset

                    runner = IterableDataset.map
                    map_kwargs = get_kwargs(map_kwargs, runner)

        return super().run(X, runner=runner, fn_kwargs=fn_kwargs, **map_kwargs)

    def _process_transform_batch_output(self, input, out, **fn_kwargs):
        bools = DataHandler.to_numpy(out).flatten().tolist() if len(out) > 0 else out[0]
        inds = [i for i, x in enumerate(bools) if x]
        if len(inds):
            return DataHandler.select_rows(input, inds)
        else:
            cols = DataHandler.get_column_names(input, generate_cols=True)
            schema = pa.schema(
                [
                    pa.field(k, string_to_arrow(v))
                    for k, v in DataHandler.get_dtypes(input).items()
                ]
            )
            return pa.Table.from_pandas(
                pd.DataFrame(columns=cols), preserve_index=False, schema=schema
            )

    def _process_transform_output(self, output, *args, **kwargs):
        init_num_rows = DataHandler.get_shape(args[0])[0]
        final_num_rows = DataHandler.get_shape(output)[0]
        if final_num_rows != init_num_rows:
            logger.info(
                f'"{self.__class__.__name__}": Selected {final_num_rows} out '
                f"of {init_num_rows} samples"
            )
        else:
            logger.info(f'"{self.__class__.__name__}": no samples were removed')
        return output
