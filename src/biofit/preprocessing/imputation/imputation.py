from dataclasses import dataclass, field

from biofit.processing import BaseProcessor, ProcessorConfig


@dataclass
class ImputationConfig(ProcessorConfig):
    """Base class for imputation processor configurations."""

    processor_name: str = field(default="imputation", init=False, repr=False)


class Imputation(BaseProcessor):
    """Base class for imputation processors."""
