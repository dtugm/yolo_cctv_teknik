"""Renderer for drawing detection results, tracking, and UI on frames."""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time

from src.config.settings import VisualizationConfig
from src.utils.geometry import intersect, get_direction, estimate_speed
from src.utils.visualization import compute_color_for_labels, UI_box


class DetectionRenderer:
    """Handles all visualization and UI rendering for detection results."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the detection renderer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Tracking data for trails
        self.data_deque: Dict[int, deque] = {}
        self.speed_line_queue: Dict[int, List[int]] = {}
        
        # Counters for objects crossing the line
        self.object_counter_in: Dict[str, int] = {}
        self.object_counter_out: Dict[str, int] = {}
        
        # Performance metrics
        self.total_detections = 0
        self.session_start_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_processing_times = deque(maxlen=30)
        
    def draw_detections(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        identities: np.ndarray,
        class_ids: np.ndarray,
        class_names: Dict[int, str]
    ) -> np.ndarray:
        """
        Draw all detections, tracking trails, and UI elements on the frame.
        
        Args:
            frame: Input frame
            bboxes: Bounding boxes in xyxy format [N, 4]
            identities: Track IDs [N]
            class_ids: Class IDs [N]
            class_names: Mapping from class ID to name
            
        Returns:
            Annotated frame
        """
        height, width, _ = frame.shape
        
        # Calculate scaled counting line based on image dimensions
        line_scaled = [
            (int(width * self.config.counting_line_start[0]), 
             int(height * self.config.counting_line_start[1])),
            (int(width * self.config.counting_line_end[0]), 
             int(height * self.config.counting_line_end[1]))
        ]
        
        # Draw counting line
        line_thickness = max(3, int(width / 800))
        cv2.line(frame, line_scaled[0], line_scaled[1], (46, 162, 112), line_thickness)
        
        # Remove old tracks from deque
        active_ids = set(identities) if len(identities) > 0 else set()
        for key in list(self.data_deque.keys()):
            if key not in active_ids:
                self.data_deque.pop(key)
                
        # Process each detection
        current_detections = 0
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
            
            track_id = int(identities[i]) if i < len(identities) else 0
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            current_detections += 1
            
            # Initialize deque for new tracks
            if track_id not in self.data_deque:
                self.data_deque[track_id] = deque(maxlen=self.config.trail_length)
                self.speed_line_queue[track_id] = []
                
            color = compute_color_for_labels(class_id)
            obj_name = class_names.get(class_id, "unknown")
            label = f'ID:{track_id} | {obj_name}'
            
            # Add center to tracking deque
            self.data_deque[track_id].appendleft(center)
            
            # Calculate speed and direction if we have history
            if len(self.data_deque[track_id]) >= 2:
                direction = get_direction(
                    self.data_deque[track_id][0], 
                    self.data_deque[track_id][1]
                )
                
                if self.config.show_speed:
                    object_speed = estimate_speed(
                        self.data_deque[track_id][1],
                        self.data_deque[track_id][0],
                        self.config.pixels_per_meter,
                        self.config.time_constant
                    )
                    self.speed_line_queue[track_id].append(object_speed)
                
                # Check if crossed counting line
                if self.config.show_counters and intersect(
                    self.data_deque[track_id][0],
                    self.data_deque[track_id][1],
                    line_scaled[0],
                    line_scaled[1]
                ):
                    cv2.line(frame, line_scaled[0], line_scaled[1], (255, 255, 255), line_thickness)
                    
                    # Determine enter/exit based on counter_direction setting
                    if self.config.counter_direction == "north_enter":
                        # North = going up = Enter, South = going down = Exit
                        if "North" in direction:
                            if obj_name not in self.object_counter_in:
                                self.object_counter_in[obj_name] = 1
                            else:
                                self.object_counter_in[obj_name] += 1
                        elif "South" in direction:
                            if obj_name not in self.object_counter_out:
                                self.object_counter_out[obj_name] = 1
                            else:
                                self.object_counter_out[obj_name] += 1
                    elif self.config.counter_direction == "south_enter":
                        # South = going down = Enter, North = going up = Exit
                        if "South" in direction:
                            if obj_name not in self.object_counter_in:
                                self.object_counter_in[obj_name] = 1
                            else:
                                self.object_counter_in[obj_name] += 1
                        elif "North" in direction:
                            if obj_name not in self.object_counter_out:
                                self.object_counter_out[obj_name] = 1
                            else:
                                self.object_counter_out[obj_name] += 1
            
            # Add speed and direction to label
            try:
                if self.config.show_speed and len(self.speed_line_queue[track_id]) > 0:
                    avg_speed = sum(self.speed_line_queue[track_id]) // len(self.speed_line_queue[track_id])
                    label += f" | {avg_speed}km/h"
                    
                if self.config.show_direction and len(self.data_deque[track_id]) >= 2:
                    label += f" | {direction}"
            except:
                pass
                
            # Draw bounding box
            UI_box(
                box, frame, label=label, color=color, 
                line_thickness=max(2, int(width / 1000))
            )
            
            # Draw tracking trail
            if self.config.show_trails:
                for j in range(1, len(self.data_deque[track_id])):
                    if self.data_deque[track_id][j - 1] is None or self.data_deque[track_id][j] is None:
                        continue
                    thickness = max(1, int(np.sqrt(64 / float(j + j)) * (width / 1000)))
                    cv2.line(
                        frame, 
                        self.data_deque[track_id][j - 1], 
                        self.data_deque[track_id][j], 
                        color, thickness
                    )
        
        self.total_detections += current_detections
        
        # Draw counters and info panel
        if self.config.show_counters:
            self._draw_counters(frame, width, height)
        self._draw_info_panel(frame, width, height)
        
        return frame
    
    def _draw_counters(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw entry/exit counters on the frame."""
        font_scale = max(0.6, width / 2500)
        thickness = max(2, int(width / 1000))
        bottom_margin = int(height * 0.25)
        panel_y = height - bottom_margin
        panel_width = int(width * 0.3)
        
        # Draw "Keluar" (Exit) panel
        entering_panel_x = width - panel_width - 20
        cv2.rectangle(
            frame, 
            (entering_panel_x, panel_y - 40), 
            (entering_panel_x + panel_width, panel_y), 
            (85, 45, 255), -1
        )
        cv2.putText(
            frame, 'Keluar', 
            (entering_panel_x + 10, panel_y - 10), 
            0, font_scale, [225, 255, 255], 
            thickness=thickness, lineType=cv2.LINE_AA
        )
        
        if self.object_counter_out:
            for idx, (key, value) in enumerate(self.object_counter_out.items()):
                cnt_str = f"{key}: {value}"
                y_pos = panel_y + 40 + (idx * 50)
                cv2.rectangle(
                    frame, 
                    (entering_panel_x, y_pos - 25), 
                    (entering_panel_x + int(panel_width * 0.8), y_pos + 15), 
                    (85, 45, 255), -1
                )
                cv2.putText(
                    frame, cnt_str, 
                    (entering_panel_x + 10, y_pos), 
                    0, font_scale, [225, 255, 255], 
                    thickness=thickness, lineType=cv2.LINE_AA
                )
        
        # Draw "Masuk" (Entry) panel
        cv2.rectangle(
            frame, 
            (20, panel_y - 40), 
            (20 + panel_width, panel_y), 
            (85, 45, 255), -1
        )
        cv2.putText(
            frame, 'Masuk', 
            (30, panel_y - 10), 
            0, font_scale, [225, 255, 255], 
            thickness=thickness, lineType=cv2.LINE_AA
        )
        
        if self.object_counter_in:
            for idx, (key, value) in enumerate(self.object_counter_in.items()):
                cnt_str = f"{key}: {value}"
                y_pos = panel_y + 40 + (idx * 50)
                cv2.rectangle(
                    frame, 
                    (20, y_pos - 25), 
                    (20 + int(panel_width * 0.8), y_pos + 15), 
                    (85, 45, 255), -1
                )
                cv2.putText(
                    frame, cnt_str, 
                    (30, y_pos), 
                    0, font_scale, [225, 255, 255], 
                    thickness=thickness, lineType=cv2.LINE_AA
                )
    
    def _draw_info_panel(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw system information panel."""
        session_duration = time.time() - self.session_start_time
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)
        seconds = int(session_duration % 60)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        avg_latency = 0
        if len(self.frame_processing_times) > 0:
            avg_latency = sum(self.frame_processing_times) / len(self.frame_processing_times) * 1000
        
        panel_height = int(height * 0.15)
        panel_width = int(width * 0.35)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, width / 3000)
        thickness = max(1, int(width / 1500))
        color = (255, 255, 255)
        y_offset = int(height * 0.03)
        line_spacing = int(height * 0.02)
        
        info_lines = [
            f"LIVE CCTV MONITORING - Resolution: {width}x{height}",
            f"Time: {current_time}",
            f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}",
            f"FPS: {self.current_fps:.1f} | Latency: {avg_latency:.1f}ms",
            f"Total Detections: {self.total_detections}",
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = y_offset + (i * line_spacing)
            cv2.putText(
                frame, line, (20, y_pos), font, font_scale, 
                color, thickness, cv2.LINE_AA
            )
    
    def update_fps(self, processing_time: float) -> None:
        """
        Update FPS calculation.
        
        Args:
            processing_time: Time taken to process current frame
        """
        self.frame_processing_times.append(processing_time)
        self.fps_counter += 1
        
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def reset(self) -> None:
        """Reset all counters and tracking data."""
        self.data_deque.clear()
        self.speed_line_queue.clear()
        self.object_counter_in.clear()
        self.object_counter_out.clear()
        self.total_detections = 0
        self.session_start_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_processing_times.clear()

