"""Configuration management for YOLO CCTV system."""

from .settings import InferenceConfig, TrackingConfig, VisualizationConfig, StreamingConfig

__all__ = [
    'InferenceConfig',
    'TrackingConfig', 
    'VisualizationConfig',
    'StreamingConfig'
]

