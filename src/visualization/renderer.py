"""Renderer for drawing detection results, tracking, and UI on frames."""

import cv2
import numpy as np
from collections import deque
from datetime import datetime, timedelta
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
        
        # Counters for objects crossing the line (only track specific classes)
        # Initialize with all tracked classes at 0
        self.tracked_classes = {"car", "person", "motorcycle"}
        self.object_counter_in: Dict[str, int] = {cls: 0 for cls in self.tracked_classes}
        self.object_counter_out: Dict[str, int] = {cls: 0 for cls in self.tracked_classes}
        
        # Reset functionality
        self.last_reset_date = datetime.now().date()
        self.auto_reset_enabled = self.config.enable_auto_daily_reset
        self.keyboard_reset_enabled = self.config.enable_keyboard_reset
        
        # Performance metrics
        self.total_detections = 0
        self.session_start_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_processing_times = deque(maxlen=30)
    
    def _check_daily_reset(self) -> None:
        """Check if a new day has started and reset counters if auto-reset is enabled."""
        current_date = datetime.now().date()
        if self.auto_reset_enabled and current_date > self.last_reset_date:
            self.reset_counters()
            self.last_reset_date = current_date
    
    def reset_counters(self) -> None:
        """Reset only the counters while keeping tracking data."""
        self.object_counter_in = {cls: 0 for cls in self.tracked_classes}
        self.object_counter_out = {cls: 0 for cls in self.tracked_classes}
        self.total_detections = 0
        
    def handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input for reset functionality.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            True if key was handled, False otherwise
        """
        if self.keyboard_reset_enabled and key == ord('r'):
            self.reset_counters()
            return True
        return False
        
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
        # Check for daily reset
        self._check_daily_reset()
        
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
            
            # Get object name and check if it's a tracked class
            obj_name = class_names.get(class_id, "unknown").lower()
            is_tracked_class = obj_name in self.tracked_classes
            
            # Initialize deque for new tracks
            if track_id not in self.data_deque:
                self.data_deque[track_id] = deque(maxlen=self.config.trail_length)
                self.speed_line_queue[track_id] = []
                
            color = compute_color_for_labels(class_id)
            label = f'ID:{track_id} | {obj_name.title()}'
            
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
                
                # Check if crossed counting line (only for tracked classes)
                if self.config.show_counters and is_tracked_class and intersect(
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
                            self.object_counter_in[obj_name] += 1
                        elif "South" in direction:
                            self.object_counter_out[obj_name] += 1
                    elif self.config.counter_direction == "south_enter":
                        # South = going down = Enter, North = going up = Exit
                        if "South" in direction:
                            self.object_counter_in[obj_name] += 1
                        elif "North" in direction:
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
        if self.config.show_info_panel:
            self._draw_info_panel(frame, width, height)
        
        return frame
    
    def _draw_counters(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw  entry/exit counters on the frame."""
        # Calculate dimensions and positions
        font_scale = max(0.7, width / 2000)
        thickness = max(2, int(width / 800))
        panel_width = int(width * 0.28)
        
        # Calculate panel height to fit exactly 3 classes with proper spacing
        header_height = 40
        item_height = 45  # Increased from 35 for better spacing
        item_spacing = 10  # Additional spacing between items
        content_height = header_height + (3 * item_height) + (2 * item_spacing) + 20  # 3 items + spacing + padding
        panel_height = content_height
        
        # Position panels at the bottom
        panel_y = height - panel_height - 20
        left_panel_x = 20
        right_panel_x = width - panel_width - 20
        
        # Color scheme
        header_color = (30, 30, 30)  # Dark gray header
        panel_color = (45, 45, 45)   # Darker gray panel
        border_color = (100, 100, 100)  # Light gray border
        text_color = (255, 255, 255)  # White text
        accent_color = (0, 150, 255)  # Blue accent
        
        # Draw left panel (Entry)
        self._draw_counter_panel(
            frame, left_panel_x, panel_y, panel_width, panel_height,
            "ENTRY", self.object_counter_in, header_color, panel_color, 
            border_color, text_color, accent_color, font_scale, thickness
        )
        
        # Draw right panel (Exit)
        self._draw_counter_panel(
            frame, right_panel_x, panel_y, panel_width, panel_height,
            "EXIT", self.object_counter_out, header_color, panel_color,
            border_color, text_color, accent_color, font_scale, thickness
        )
        
        # Draw reset instructions
        if self.keyboard_reset_enabled:
            reset_text = "Press 'R' to reset counters"
            reset_y = panel_y - 30
            cv2.putText(
                frame, reset_text,
                (width // 2 - len(reset_text) * 8, reset_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (200, 200, 200),
                thickness, cv2.LINE_AA
            )
    
    def _draw_counter_panel(self, frame: np.ndarray, x: int, y: int, width: int, height: int,
                           title: str, counters: Dict[str, int], header_color: Tuple[int, int, int],
                           panel_color: Tuple[int, int, int], border_color: Tuple[int, int, int],
                           text_color: Tuple[int, int, int], accent_color: Tuple[int, int, int],
                           font_scale: float, thickness: int) -> None:
        """Draw a single counter panel with styling."""
        # Draw main panel with border
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2)
        cv2.rectangle(frame, (x + 2, y + 2), (x + width - 2, y + height - 2), panel_color, -1)
        
        # Draw header
        header_height = 40
        cv2.rectangle(frame, (x + 2, y + 2), (x + width - 2, y + header_height), header_color, -1)
        
        # Draw title
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        title_x = x + (width - title_size[0]) // 2
        title_y = y + header_height - 10
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, accent_color, thickness, cv2.LINE_AA)
        
        # Draw counters - always show all three tracked classes
        item_height = 45
        item_spacing = 10
        start_y = y + header_height + 15
        
        # Define the order of classes to display
        class_order = ["car", "person", "motorcycle"]
        
        for idx, obj_name in enumerate(class_order):
            count = counters.get(obj_name, 0)
            
            # Draw counter item background
            item_y = start_y + (idx * (item_height + item_spacing))
            cv2.rectangle(frame, (x + 8, item_y - 20), (x + width - 8, item_y + 10), 
                         (60, 60, 60), -1)
            
            # Draw object name
            obj_text = obj_name.title()
            cv2.putText(frame, obj_text, (x + 15, item_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale * 0.8, text_color, thickness, cv2.LINE_AA)
            
            # Draw count with accent color
            count_text = str(count)
            count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            count_x = x + width - count_size[0] - 15
            cv2.putText(frame, count_text, (count_x, item_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, accent_color, thickness, cv2.LINE_AA)
    
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
        self.reset_counters()
        self.session_start_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_processing_times.clear()

