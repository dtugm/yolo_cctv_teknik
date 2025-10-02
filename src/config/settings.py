"""Configuration classes for the YOLO CCTV inference system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class InferenceConfig:
    """Configuration for YOLO model inference."""
    
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_detections: int = 300
    image_size: int = 640
    device: str = "cuda"  # "cuda" or "cpu"
    half_precision: bool = False
    
    
@dataclass
class TrackingConfig:
    """Configuration for DeepSORT tracking."""
    
    # Note: reid_checkpoint path is relative to deep_sort_pytorch directory
    reid_checkpoint: str = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    max_dist: float = 0.2
    min_confidence: float = 0.3
    nms_max_overlap: float = 0.5
    max_iou_distance: float = 0.7
    max_age: int = 70
    n_init: int = 3
    nn_budget: int = 100
    use_cuda: bool = True
    

@dataclass
class VisualizationConfig:
    """Configuration for visualization and UI elements."""
    
    line_thickness: int = 0.5
    font_scale: float = 0.1
    show_trails: bool = False
    trail_length: int = 64
    show_speed: bool = True
    show_direction: bool = True
    show_counters: bool = False
    
    # Counting line coordinates (as fraction of image size)
    counting_line_start: Tuple[float, float] = (0.1, 0.6)  # (x, y) as fraction
    counting_line_end: Tuple[float, float] = (0.9, 0.6)
    
    # Speed estimation parameters
    pixels_per_meter: int = 10
    time_constant: float = 15 * 3.6  # Convert to km/h
    

@dataclass
class StreamingConfig:
    """Configuration for HTTP streaming server."""
    
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 5050
    url_prefix: str = "/_yolo_stream"
    jpeg_quality: int = 80
    target_fps: int = 30

