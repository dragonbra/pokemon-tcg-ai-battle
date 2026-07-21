"""Observation and legal-option encoders for the official PTCG observation shape."""

from .features import FEATURE_SCHEMA_VERSION, PTCGFeatureConfig, encode_observation
from .dataset import (
    DATASET_VERSION,
    load_behavior_cloning_dataset,
    write_behavior_cloning_dataset,
)

__all__ = [
    "DATASET_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "PTCGFeatureConfig",
    "encode_observation",
    "load_behavior_cloning_dataset",
    "write_behavior_cloning_dataset",
]
