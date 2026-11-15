"""
Engine module initialization.
"""

from src.engine.evaluator import (
    compute_category_sbir_metrics_np,
    compute_fine_grained_sbir_metrics_np,
    AdaptiveOODDetector,
)
from src.engine.trainer import (
    extract_features_dataloader,
    precompute_photo_cache,
    run_training_pipeline,
)

__all__ = [
    "compute_category_sbir_metrics_np",
    "compute_fine_grained_sbir_metrics_np",
    "AdaptiveOODDetector",
    "extract_features_dataloader",
    "precompute_photo_cache",
    "run_training_pipeline",
]
