import os
import sys
from typing import TYPE_CHECKING, Optional

import joblib
from biocore.utils.import_util import is_optuna_available

if TYPE_CHECKING:
    import optuna


def report_generation(data, study: Optional["optuna.Study"] = None, cache_dir=None):
    if cache_dir is None and (
        (
            "biosets" in sys.modules
            and isinstance(data, getattr(sys.modules["biosets"], "Bioset"))
        )
        or (
            "datasets" in sys.modules
            and isinstance(data, getattr(sys.modules["datasets"], "Dataset"))
        )
    ):
        cache_dir = os.path.dirname(data.cache_files[0]["filename"])

    if study is None:
        if cache_dir:
            with open(os.path.join(cache_dir, "study.joblib"), "rb") as f:
                study = joblib.load(f)

    if is_optuna_available() and study:
        import optuna.visualization as ov

        if study is None:
            raise ValueError("Study object is required for optuna report generation")
        ov.plot_optimization_history(study)
