from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
from biocore import DataHandler

from biofit.processing import BaseProcessor, ProcessorConfig
from biofit.utils import logging
from biofit.utils.table_util import string_to_arrow

logger = logging.get_logger(__name__)


@dataclass
class ResamplerConfig(ProcessorConfig):
    processor_type: str = field(default="resampling", init=False, repr=False)


class Resampler(BaseProcessor):
    _feature_dependent = (
        False  # SampleFiltering does not depend on features when transforming
    )

    def _process_transform_batch_output(self, input, out, **fn_kwargs):
        inds = out["indices"]
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
                f'"{self.__class__.__name__}": Resampled to {final_num_rows} from '
                f"{init_num_rows} samples"
            )
        else:
            logger.info(f'"{self.__class__.__name__}": No resampling performed')
        return output

    pass
