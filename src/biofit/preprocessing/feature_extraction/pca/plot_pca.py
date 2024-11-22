from dataclasses import dataclass, field
from typing import Optional

from biofit.utils.types import Unset
from biofit.visualization.plotting_utils import plot_dimension_reduction

from ..plot_feature_extraction import (
    FeatureExtractorPlotter,
    FeatureExtractorPlotterConfig,
)


@dataclass
class PCAFeatureExtractorPlotterConfig(FeatureExtractorPlotterConfig):
    processor_name: str = field(default="pca", init=False, repr=False)
    title: str = "PCA Plot"
    n_components: int = 3
    label_column: str = None
    group_column: str = None


class PCAFeatureExtractorPlotter(FeatureExtractorPlotter):
    _config_class = PCAFeatureExtractorPlotterConfig
    config: PCAFeatureExtractorPlotterConfig

    def __init__(
        self,
        n_components: int = 3,
        label_column: str = None,
        group_column: str = None,
        config: Optional[PCAFeatureExtractorPlotterConfig] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            n_components=n_components,
            label_column=label_column,
            group_column=group_column,
            **kwargs,
        )

    def plot(
        self,
        X,
        labels=None,
        group=None,
        input_columns=None,
        label_column=None,
        group_column=None,
        precomputed: bool = Unset("False"),
        pca_kwargs: dict = Unset("{}"),
        title: str = Unset("PCA Plot"),
        n_components: int = Unset(3),
        path=None,
        **kwargs,
    ):
        """Plot the PCA plot.

        Args:
            X: The input data.
            labels: The labels for the data.
            group: The group for the data.
            input_columns: The input columns.
            label_column: The label column.
            group_column: The group column.
            precomputed: Whether the input data is precomputed.
            pca_kwargs: The additional kwargs for computing PCA. Only used if precomputed is False.
            **kwargs: The additional kwargs.

        """
        return plot_dimension_reduction(
            X,
            labels=labels,
            group=group,
            input_columns=input_columns,
            label_column=label_column,
            group_column=group_column,
            method="pca" if not precomputed else None,
            method_kwargs=pca_kwargs,
            output_dir=path,
            **kwargs,
        )
