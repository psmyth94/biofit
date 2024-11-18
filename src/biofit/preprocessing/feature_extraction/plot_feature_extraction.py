from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from biofit.visualization.plotting import BasePlotter, PlotterConfig

if TYPE_CHECKING:
    pass


@dataclass
class FeatureExtractorPlotterConfig(PlotterConfig):
    processor_type: str = field(default="feature_extractor", init=False, repr=False)


class FeatureExtractorPlotter(BasePlotter):
    """Base class for feature extraction processors."""

    config_class = FeatureExtractorPlotterConfig
    config: FeatureExtractorPlotterConfig
