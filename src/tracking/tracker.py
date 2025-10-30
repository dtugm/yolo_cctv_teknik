"""Object tracker using DeepSORT algorithm with speed estimation."""

import sys
from pathlib import Path
import torch
from typing import Optional, Tuple, List, Dict
import numpy as np
from collections import defaultdict
import time

# Add deep_sort_pytorch to path
DEEP_SORT_PATH = Path(__file__).parent.parent.parent / "ultralytics" / "yolo" / "v8" / "detect"
sys.path.insert(0, str(DEEP_SORT_PATH))

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from src.config.settings import TrackingConfig


class TrackedObject:
    """Represents a tracked object with speed estimation."""
    
    def __init__(self, track_id: int, bbox: np.ndarray, class_id: int):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.speed = 0.0  # km/h
        self.positions = []  # List of (timestamp, center_y) tuples
        self.max_history = 30  # Keep last 30 positions
        
    def update_position(self, bbox: np.ndarray, timestamp: float):
        """Update position and calculate speed."""
        self.bbox = bbox
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        self.positions.append((timestamp, center_y))
        
        # Keep only recent history
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
    
    def calculate_speed(self, pixels_per_meter: float, fps: float) -> float:
        """
        Calculate speed based on position history.
        
        Args:
            pixels_per_meter: Calibration factor (pixels per meter)
            fps: Video frame rate
            
        Returns:
            Speed in km/h
        """
        if len(self.positions) < 2:
            self.speed = 0.0
            return 0.0
        
        # Use first and last position for speed calculation
        time_start, pos_start = self.positions[0]
        time_end, pos_end = self.positions[-1]
        
        time_diff = time_end - time_start
        
        if time_diff <= 0:
            self.speed = 0.0
            return 0.0
        
        # Calculate pixel displacement
        pixel_distance = abs(pos_end - pos_start)
        
        # Convert to meters
        distance_meters = pixel_distance / pixels_per_meter
        
        # Calculate speed in m/s
        speed_ms = distance_meters / time_diff
        
        # Convert to km/h
        self.speed = speed_ms * 3.6
        
        return self.speed
    
    def to_tlbr(self) -> np.ndarray:
        """Get bounding box in [x1, y1, x2, y2] format."""
        return self.bbox
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (has enough history)."""
        return len(self.positions) >= 3


class ObjectTracker:
    """Wrapper for DeepSORT object tracking with speed estimation."""
    
    def __init__(
        self, 
        config: Optional[TrackingConfig] = None,
        pixels_per_meter: float = 20.0,  # Calibration: pixels per meter
        fps: float = 30.0  # Video frame rate
    ):
        """
        Initialize object tracker with speed estimation.
        
        Args:
            config: Tracking configuration. Uses defaults if None.
            pixels_per_meter: Calibration factor for speed calculation
            fps: Video frame rate
        """
        self.config = config or TrackingConfig()
        self.tracker: Optional[DeepSort] = None
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        
        # Track management
        self.tracks: Dict[int, TrackedObject] = {}
        self.frame_count = 0
        self.start_time = time.time()
        
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
            str(reid_ckpt_path),
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
        Update tracker with new detections and calculate speeds.
        
        Args:
            bboxes: Bounding boxes in xywh format [N, 4]
            confidences: Detection confidences [N, 1]
            class_ids: Class IDs for each detection
            frame: Current frame (for feature extraction)
            
        Returns:
            Tracked outputs with format [x1, y1, x2, y2, track_id, class_id]
        """
        self.frame_count += 1
        current_time = time.time()
        
        if len(bboxes) == 0:
            return np.array([])
        
        # Update DeepSORT tracker
        outputs = self.tracker.update(bboxes, confidences, class_ids, frame)
        
        if len(outputs) == 0:
            return outputs
        
        # Update tracked objects with speed estimation
        active_track_ids = set()
        
        for output in outputs:
            x1, y1, x2, y2, track_id, class_id = output
            track_id = int(track_id)
            class_id = int(class_id)
            
            active_track_ids.add(track_id)
            bbox = np.array([x1, y1, x2, y2])
            
            # Create or update tracked object
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackedObject(track_id, bbox, class_id)
                print(f"🆕 New track created: ID {track_id}")
            
            # Update position and calculate speed
            self.tracks[track_id].update_position(bbox, current_time)
            speed = self.tracks[track_id].calculate_speed(
                self.pixels_per_meter, 
                self.fps
            )
            
            # Debug: Print speed for first few frames of each track
            if len(self.tracks[track_id].positions) <= 5:
                print(f"   Track {track_id}: {len(self.tracks[track_id].positions)} positions, speed: {speed:.1f} km/h")
        
        # Remove inactive tracks
        inactive_ids = set(self.tracks.keys()) - active_track_ids
        for track_id in inactive_ids:
            del self.tracks[track_id]
        
        # Print summary every 100 frames
        if self.frame_count % 100 == 0:
            print(f"\n📊 Frame {self.frame_count}: {len(self.tracks)} active tracks")
            for track_id, track in self.tracks.items():
                print(f"   Track {track_id}: {track.speed:.1f} km/h ({len(track.positions)} positions)")
        
        return outputs
    
    def get_track_speed(self, track_id: int) -> Optional[float]:
        """Get speed of a specific track."""
        if track_id in self.tracks:
            return self.tracks[track_id].speed
        return None
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracks.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self._initialize_tracker()
