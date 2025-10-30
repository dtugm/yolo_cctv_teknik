#!/usr/bin/env python3
"""
Plate capture manager for speed violation detection.
Exports bounding box images of vehicles exceeding speed limits.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import json


class PlateCaptureManager:
    """
    Manages capture of vehicle images when speed violations are detected.
    """
    
    def __init__(
        self,
        output_dir: str = "output/violations",
        speed_limit: float = 35.0,  # km/h
        enabled: bool = True,
        save_metadata: bool = True,
        image_format: str = "jpg",
        image_quality: int = 95,
        min_bbox_area: int = 1000,  # Minimum bounding box area to capture
    ):
        """
        Initialize the plate capture manager.
        
        Args:
            output_dir: Directory to save violation images
            speed_limit: Speed limit threshold in km/h
            enabled: Enable/disable capture functionality
            save_metadata: Save JSON metadata with each capture
            image_format: Image format (jpg, png)
            image_quality: JPEG quality (1-100)
            min_bbox_area: Minimum bounding box area in pixels
        """
        self.output_dir = Path(output_dir)
        self.speed_limit = speed_limit
        self.enabled = enabled
        self.save_metadata = save_metadata
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.min_bbox_area = min_bbox_area
        
        # Statistics
        self.total_violations = 0
        self.total_captures = 0
        self.failed_captures = 0
        
        # Create output directory
        if self.enabled:
            self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        print(f"📁 Plate capture output directory: {self.output_dir.absolute()}")
    
    def check_and_capture(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        speed: float,  # km/h
        class_name: str = "vehicle",
        timestamp: Optional[datetime] = None,
        additional_info: Optional[dict] = None
    ) -> Optional[str]:
        """
        Check if vehicle exceeds speed limit and capture if violation detected.
        
        Args:
            frame: Current video frame
            track_id: Unique tracking ID
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            speed: Detected speed in km/h
            class_name: Object class name
            timestamp: Capture timestamp (auto-generated if None)
            additional_info: Additional metadata to save
        
        Returns:
            Path to saved image if captured, None otherwise
        """
        if not self.enabled:
            return None
        
        # Check if speed exceeds limit
        if speed <= self.speed_limit:
            return None
        
        self.total_violations += 1
        
        # Validate bounding box
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if bbox_area < self.min_bbox_area:
            print(f"⚠️  Bbox too small for track {track_id}: {bbox_area}px²")
            self.failed_captures += 1
            return None
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️  Invalid bbox for track {track_id}")
            self.failed_captures += 1
            return None
        
        # Extract vehicle region
        vehicle_crop = frame[y1:y2, x1:x2].copy()
        
        if vehicle_crop.size == 0:
            print(f"⚠️  Empty crop for track {track_id}")
            self.failed_captures += 1
            return None
        
        # Generate filename
        if timestamp is None:
            timestamp = datetime.now()
        
        filename = self._generate_filename(track_id, speed, timestamp)
        
        # Save image
        image_path = self.output_dir / "images" / filename
        success = self._save_image(vehicle_crop, image_path)
        
        if not success:
            self.failed_captures += 1
            return None
        
        # Save metadata
        if self.save_metadata:
            metadata = self._create_metadata(
                track_id=track_id,
                bbox=bbox,
                speed=speed,
                class_name=class_name,
                timestamp=timestamp,
                image_path=str(image_path),
                additional_info=additional_info
            )
            self._save_metadata(metadata, filename)
        
        self.total_captures += 1
        print(f"🚨 VIOLATION: Track {track_id} @ {speed:.1f} km/h → {image_path.name}")
        
        return str(image_path)
    
    def _generate_filename(
        self,
        track_id: int,
        speed: float,
        timestamp: datetime
    ) -> str:
        """Generate unique filename for captured image."""
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        speed_str = f"{speed:.1f}".replace(".", "_")
        filename = f"violation_track{track_id}_{speed_str}kmh_{timestamp_str}.{self.image_format}"
        return filename
    
    def _save_image(self, image: np.ndarray, path: Path) -> bool:
        """Save image to disk."""
        try:
            if self.image_format == "jpg" or self.image_format == "jpeg":
                cv2.imwrite(
                    str(path),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
                )
            elif self.image_format == "png":
                cv2.imwrite(
                    str(path),
                    image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9]
                )
            else:
                cv2.imwrite(str(path), image)
            
            return True
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False
    
    def _create_metadata(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        speed: float,
        class_name: str,
        timestamp: datetime,
        image_path: str,
        additional_info: Optional[dict] = None
    ) -> dict:
        """Create metadata dictionary."""
        metadata = {
            "track_id": track_id,
            "speed_kmh": round(speed, 2),
            "speed_limit_kmh": self.speed_limit,
            "speed_over_limit_kmh": round(speed - self.speed_limit, 2),
            "class": class_name,
            "bbox": {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
                "width": int(bbox[2] - bbox[0]),
                "height": int(bbox[3] - bbox[1])
            },
            "timestamp": timestamp.isoformat(),
            "image_path": image_path,
            "violation_number": self.total_captures + 1
        }
        
        if additional_info:
            metadata["additional_info"] = additional_info
        
        return metadata
    
    def _save_metadata(self, metadata: dict, image_filename: str):
        """Save metadata as JSON file."""
        try:
            json_filename = Path(image_filename).stem + ".json"
            json_path = self.output_dir / "metadata" / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving metadata: {e}")
    
    def get_statistics(self) -> dict:
        """Get capture statistics."""
        return {
            "total_violations_detected": self.total_violations,
            "total_captures_saved": self.total_captures,
            "failed_captures": self.failed_captures,
            "success_rate": (
                self.total_captures / self.total_violations * 100
                if self.total_violations > 0 else 0
            )
        }
    
    def print_statistics(self):
        """Print capture statistics."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("📊 PLATE CAPTURE STATISTICS")
        print("="*50)
        print(f"Speed Limit: {self.speed_limit} km/h")
        print(f"Total Violations Detected: {stats['total_violations_detected']}")
        print(f"Total Captures Saved: {stats['total_captures_saved']}")
        print(f"Failed Captures: {stats['failed_captures']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Output Directory: {self.output_dir.absolute()}")
        print("="*50 + "\n")
