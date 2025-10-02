"""YOLO CCTV Inference System - Refactored and Modular."""

__version__ = "2.0.0"

from .core import InferenceEngine
from .config import InferenceConfig, TrackingConfig, VisualizationConfig, StreamingConfig
from .streaming import StreamingServer

__all__ = [
    'InferenceEngine',
    'InferenceConfig',
    'TrackingConfig',
    'VisualizationConfig',
    'StreamingConfig',
    'StreamingServer'
]

