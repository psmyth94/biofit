from dataclasses import dataclass, field
from pathlib import Path

from biofit.visualization.plotting import BasePlotter, PlotterConfig


@dataclass
class SampleFilterPlotterConfig(PlotterConfig):
    processor_type: str = field(default="filtering", init=False, repr=False)
    r_source: str = field(
        default=(Path(__file__).parent / "plot_filtering.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="plot_filter", init=False, repr=False)


class SampleFilterPlotter(BasePlotter):
    config_class = SampleFilterPlotterConfig
    config: SampleFilterPlotterConfig
