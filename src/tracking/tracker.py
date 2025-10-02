"""Object tracker using DeepSORT algorithm."""

import sys
from pathlib import Path
import torch
from typing import Optional, Tuple, List
import numpy as np

# Add deep_sort_pytorch to path
DEEP_SORT_PATH = Path(__file__).parent.parent.parent / "ultralytics" / "yolo" / "v8" / "detect"
sys.path.insert(0, str(DEEP_SORT_PATH))

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from src.config.settings import TrackingConfig


class ObjectTracker:
    """Wrapper for DeepSORT object tracking."""
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize object tracker.
        
        Args:
            config: Tracking configuration. Uses defaults if None.
        """
        self.config = config or TrackingConfig()
        self.tracker: Optional[DeepSort] = None
        self._initialize_tracker()
        
    def _initialize_tracker(self) -> None:
        """Initialize the DeepSORT tracker with configuration."""
        # Load DeepSORT config
        cfg_deep = get_config()
        config_path = DEEP_SORT_PATH / "deep_sort_pytorch" / "configs" / "deep_sort.yaml"
        cfg_deep.merge_from_file(str(config_path))
        
        # Get absolute path to reid checkpoint
        reid_ckpt_path = DEEP_SORT_PATH / "deep_sort_pytorch" / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"
        
        # Create tracker with configured parameters
        self.tracker = DeepSort(
            str(reid_ckpt_path),  # Use absolute path
            max_dist=self.config.max_dist,
            min_confidence=self.config.min_confidence,
            nms_max_overlap=self.config.nms_max_overlap,
            max_iou_distance=self.config.max_iou_distance,
            max_age=self.config.max_age,
            n_init=self.config.n_init,
            nn_budget=self.config.nn_budget,
            use_cuda=self.config.use_cuda
        )
        
    def update(
        self, 
        bboxes: torch.Tensor, 
        confidences: torch.Tensor, 
        class_ids: List[int], 
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Update tracker with new detections.
        
        Args:
            bboxes: Bounding boxes in xywh format [N, 4]
            confidences: Detection confidences [N, 1]
            class_ids: Class IDs for each detection
            frame: Current frame (for feature extraction)
            
        Returns:
            Tracked outputs with format [x1, y1, x2, y2, track_id, class_id]
        """
        if len(bboxes) == 0:
            return np.array([])
            
        outputs = self.tracker.update(bboxes, confidences, class_ids, frame)
        return outputs
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self._initialize_tracker()

