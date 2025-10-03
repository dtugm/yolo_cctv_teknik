# live_predict.py (modified for HTTP live stream)
# User original code with streaming additions

# --- imports (original + flask) ---
import hydra
import torch
import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from datetime import datetime
import threading

# new imports for streaming
from flask import Flask, Response
import socket
import sys
import os
import json

# --- existing globals ---
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}
object_counter1 = {}

# Logging configuration
LOG_FORMAT = "json"  # "json" or "txt"
LOG_FILE_JSON = "vehicle_detection_log.json"
LOG_FILE_TXT = "vehicle_detection_log.txt"
LOG_SUMMARY_JSON = "vehicle_summary.json"

# Logging variables
log_lock = threading.Lock()
last_logged_counters = {"in": {}, "out": {}}
detection_log = []


# Enhanced reporting variables
total_detections = 0
session_start_time = time.time()
fps_counter = 0
fps_start_time = time.time()
current_fps = 0
frame_processing_times = deque(maxlen=30)

line = [(100, 500), (1050, 500)]
speed_line_queue = {}

# --- streaming globals ---
ENABLE_HTTP_STREAM = True
HTTP_PORT = 5000
http_streamer = None

# (rest of your functions unchanged...) eg. estimatespeed, init_tracker, xyxy_to_xywh, etc.
# I am pasting the user's original functions without changes for brevity in this message.
# In your actual file ensure the following functions (estimatespeed, init_tracker, xyxy_to_xywh,
# xyxy_to_tlwh, compute_color_for_labels, draw_border, UI_box, intersect, ccw, get_direction,
# calculate_latency, draw_enhanced_info_panel, draw_boxes) are present exactly as before.

# ------------------------------
# (PASTE ALL YOUR ORIGINAL HELPERS HERE)
# ------------------------------
# For readability here I will include them directly as they were in your uploaded file:
# -- begin pasted helper functions (unchanged) --

def log_vehicle_detection(direction, vehicle_type, vehicle_id, timestamp=None):
    """Log individual vehicle detection with timestamp"""
    global detection_log
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "direction": direction,  # "in" or "out"
        "vehicle_type": vehicle_type,
        "vehicle_id": vehicle_id,
        "session_time": time.time() - session_start_time
    }
    
    with log_lock:
        detection_log.append(log_entry)
        
        # Write to JSON file
        if LOG_FORMAT == "json" or LOG_FORMAT == "both":
            write_json_log(log_entry)
        
        # Write to TXT file
        if LOG_FORMAT == "txt" or LOG_FORMAT == "both":
            write_txt_log(log_entry)
    
    print(f"🚗 {direction.upper()}: {vehicle_type} (ID: {vehicle_id}) at {timestamp}")

def write_json_log(log_entry):
    """Write individual log entry to JSON file"""
    try:
        # Read existing data
        if os.path.exists(LOG_FILE_JSON):
            with open(LOG_FILE_JSON, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"detections": [], "summary": {}}
        else:
            data = {"detections": [], "summary": {}}
        
        # Add new detection
        data["detections"].append(log_entry)
        
        # Update summary
        direction = log_entry["direction"]
        vehicle_type = log_entry["vehicle_type"]
        
        if "summary" not in data:
            data["summary"] = {"in": {}, "out": {}, "total": {}}
        
        if direction not in data["summary"]:
            data["summary"][direction] = {}
        
        if vehicle_type not in data["summary"][direction]:
            data["summary"][direction][vehicle_type] = 0
        
        data["summary"][direction][vehicle_type] += 1
        
        # Update total summary
        if "total" not in data["summary"]:
            data["summary"]["total"] = {}
        
        if vehicle_type not in data["summary"]["total"]:
            data["summary"]["total"][vehicle_type] = 0
        
        data["summary"]["total"][vehicle_type] = (
            data["summary"].get("in", {}).get(vehicle_type, 0) + 
            data["summary"].get("out", {}).get(vehicle_type, 0)
        )
        
        # Add metadata
        data["metadata"] = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration": time.time() - session_start_time,
            "total_detections": len(data["detections"])
        }
        
        # Write back to file
        with open(LOG_FILE_JSON, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error writing JSON log: {e}")

def write_txt_log(log_entry):
    """Write individual log entry to TXT file"""
    try:
        log_line = f"{log_entry['timestamp']} | {log_entry['direction'].upper()} | {log_entry['vehicle_type']} | ID: {log_entry['vehicle_id']}\n"
        
        with open(LOG_FILE_TXT, 'a') as f:
            f.write(log_line)
            
    except Exception as e:
        print(f"❌ Error writing TXT log: {e}")

def update_summary_file():
    """Update summary file with current counters"""
    try:
        summary_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration": time.time() - session_start_time,
            "vehicles_in": dict(object_counter1),  # North direction (entering)
            "vehicles_out": dict(object_counter),   # South direction (exiting)
            "total_in": sum(object_counter1.values()),
            "total_out": sum(object_counter.values()),
            "net_vehicles": sum(object_counter1.values()) - sum(object_counter.values()),
            "fps": current_fps,
            "total_detections": total_detections
        }
        
        # Calculate combined totals
        all_vehicle_types = set(list(object_counter.keys()) + list(object_counter1.keys()))
        summary_data["combined_totals"] = {}
        
        for vehicle_type in all_vehicle_types:
            in_count = object_counter1.get(vehicle_type, 0)
            out_count = object_counter.get(vehicle_type, 0)
            summary_data["combined_totals"][vehicle_type] = {
                "in": in_count,
                "out": out_count,
                "total": in_count + out_count
            }
        
        with open(LOG_SUMMARY_JSON, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
    except Exception as e:
        print(f"❌ Error updating summary file: {e}")

def initialize_log_files():
    """Initialize log files with headers"""
    try:
        # Initialize JSON log
        if not os.path.exists(LOG_FILE_JSON):
            initial_data = {
                "detections": [],
                "summary": {"in": {}, "out": {}, "total": {}},
                "metadata": {
                    "session_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_detections": 0
                }
            }
            with open(LOG_FILE_JSON, 'w') as f:
                json.dump(initial_data, f, indent=2)
        
        # Initialize TXT log
        if not os.path.exists(LOG_FILE_TXT):
            with open(LOG_FILE_TXT, 'w') as f:
                f.write(f"=== Vehicle Detection Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write("Format: TIMESTAMP | DIRECTION | VEHICLE_TYPE | ID\n")
                f.write("=" * 80 + "\n")
        
        print("📝 Log files initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing log files: {e}")


def estimatespeed(Location1, Location2):
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    ppm = 15
    d_meters = d_pixel/ppm
    time_constant = 15*3.6
    speed = d_meters * time_constant
    return int(speed)

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                       max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                       min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                       nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                       max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                       max_age=cfg_deep.DEEPSORT.MAX_AGE,
                       n_init=cfg_deep.DEEPSORT.N_INIT,
                       nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                       use_cuda=True)

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    color_map = {
        0: (85, 45, 255),
        1: (255, 0, 0),
        2: (222, 82, 175),
        3: (0, 204, 255),
        4: (255, 255, 0),
        5: (0, 149, 255),
        6: (0, 255, 0),
        7: (255, 0, 255),
    }
    if label in color_map:
        return color_map[label]
    else:
        label = int(label) % 255
        color = [int((p * (label * 2 + 1)) % 255) for p in palette]
        return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.003 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        font_scale = tl / 2.5
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, font_scale, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def get_direction(point1, point2):
    direction_str = ""
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""
    return direction_str

def calculate_latency():
    if len(frame_processing_times) > 0:
        return sum(frame_processing_times) / len(frame_processing_times) * 1000
    return 0

def draw_enhanced_info_panel(img):
    global current_fps, total_detections, session_start_time
    height, width, _ = img.shape
    session_duration = time.time() - session_start_time
    hours = int(session_duration // 3600)
    minutes = int((session_duration % 3600) // 60)
    seconds = int(session_duration % 60)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_latency = calculate_latency()
    panel_height = int(height * 0.15)
    panel_width = int(width * 0.35)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
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
        f"FPS: {current_fps:.1f} | Latency: {avg_latency:.1f}ms",
        f"Total Detections: {total_detections}",
        f"RTSP Stream Active"
    ]
    for i, line in enumerate(info_lines):
        y_pos = y_offset + (i * line_spacing)
        cv2.putText(img, line, (20, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    global total_detections
    height, width, _ = img.shape
    line_scaled = [(int(width * 0.1), int(height * 0.6)), (int(width * 0.9), int(height * 0.6))]
    line_thickness = max(3, int(width / 800))
    cv2.line(img, line_scaled[0], line_scaled[1], (46, 162, 112), line_thickness)
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)
    current_detections = 0
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
        id = int(identities[i]) if identities is not None else 0
        current_detections += 1
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = f'ID:{id} | {obj_name}'
        data_deque[id].appendleft(center)
        # In the draw_boxes function, replace the intersection detection section:
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)
            
            if intersect(data_deque[id][0], data_deque[id][1], line_scaled[0], line_scaled[1]):
                cv2.line(img, line_scaled[0], line_scaled[1], (255, 255, 255), line_thickness)
                
                # Enhanced logging with direction detection
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                        # Log the vehicle going OUT
                        log_vehicle_detection("out", obj_name, id)
                    else:
                        # Check if this is a new detection for this ID
                        if id not in last_logged_counters.get("out", {}):
                            object_counter[obj_name] += 1
                            log_vehicle_detection("out", obj_name, id)
                            if "out" not in last_logged_counters:
                                last_logged_counters["out"] = {}
                            last_logged_counters["out"][id] = True
                            
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                        # Log the vehicle going IN
                        log_vehicle_detection("in", obj_name, id)
                    else:
                        # Check if this is a new detection for this ID
                        if id not in last_logged_counters.get("in", {}):
                            object_counter1[obj_name] += 1
                            log_vehicle_detection("in", obj_name, id)
                            if "in" not in last_logged_counters:
                                last_logged_counters["in"] = {}
                            last_logged_counters["in"][id] = True

        try:
            avg_speed = sum(speed_line_queue[id]) // len(speed_line_queue[id])
            label += f" | {avg_speed}km/h"
            if len(data_deque[id]) >= 2:
                label += f" | {direction}"
        except:
            if len(data_deque[id]) >= 2:
                label += f" | {direction}"
        UI_box(box, img, label=label, color=color, line_thickness=max(2, int(width/1000)))
        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            thickness = max(1, int(np.sqrt(64 / float(j + j)) * (width / 1000)))
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness)
    total_detections += current_detections
    font_scale = max(0.6, width / 2500)
    thickness = max(2, int(width / 1000))
    bottom_margin = int(height * 0.25)
    panel_y = height - bottom_margin
    panel_width = int(width * 0.3)
    entering_panel_x = width - panel_width - 20
    cv2.rectangle(img, (entering_panel_x, panel_y - 40), (entering_panel_x + panel_width, panel_y), (85, 45, 255), -1)
    cv2.putText(img, 'Keluar', (entering_panel_x + 10, panel_y - 10), 0, font_scale, [225, 255, 255], thickness=thickness, lineType=cv2.LINE_AA)
    if object_counter1:
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = f"{key}: {value}"
            y_pos = panel_y + 40 + (idx * 50)
            cv2.rectangle(img, (entering_panel_x, y_pos - 25), (entering_panel_x + int(panel_width * 0.8), y_pos + 15), (85, 45, 255), -1)
            cv2.putText(img, cnt_str, (entering_panel_x + 10, y_pos), 0, font_scale, [225, 255, 255], thickness=thickness, lineType=cv2.LINE_AA)
    else:
        y_pos = panel_y + 40
        cv2.rectangle(img, (entering_panel_x, y_pos - 25), (entering_panel_x + int(panel_width * 0.8), y_pos + 15), (85, 45, 255), -1)
        cv2.putText(img, "No vehicles detected", (entering_panel_x + 10, y_pos), 0, font_scale * 0.8, [200, 200, 200], thickness=max(1, thickness-1), lineType=cv2.LINE_AA)
    cv2.rectangle(img, (20, panel_y - 40), (20 + panel_width, panel_y), (85, 45, 255), -1)
    cv2.putText(img, 'Masuk', (30, panel_y - 10), 0, font_scale, [225, 255, 255], thickness=thickness, lineType=cv2.LINE_AA)
    if object_counter:
        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str = f"{key}: {value}"
            y_pos = panel_y + 40 + (idx * 50)
            cv2.rectangle(img, (20, y_pos - 25), (20 + int(panel_width * 0.8), y_pos + 15), (85, 45, 255), -1)
            cv2.putText(img, cnt_str, (30, y_pos), 0, font_scale, [225, 255, 255], thickness=thickness, lineType=cv2.LINE_AA)
    else:
        y_pos = panel_y + 40
        cv2.rectangle(img, (20, y_pos - 25), (20 + int(panel_width * 0.8), y_pos + 15), (85, 45, 255), -1)
        cv2.putText(img, "No vehicles detected", (30, y_pos), 0, font_scale * 0.8, [200, 200, 200], thickness=max(1, thickness-1), lineType=cv2.LINE_AA)
    draw_enhanced_info_panel(img)
    return img

def periodic_summary_update():
    """Periodically update summary file every 5 seconds"""
    while True:
        time.sleep(5)  # Update every 5 seconds
        update_summary_file()

# Start the periodic update thread
def start_logging_services():
    """Start all logging services"""
    initialize_log_files()
    
    # Start periodic summary update thread
    summary_thread = threading.Thread(target=periodic_summary_update, daemon=True)
    summary_thread.start()
    
    print("📊 Logging services started")
    print(f"📝 JSON Log: {LOG_FILE_JSON}")
    print(f"📝 TXT Log: {LOG_FILE_TXT}")
    print(f"📊 Summary: {LOG_SUMMARY_JSON}")


# -- end pasted helper functions --

# --- HTTP Streamer class ---
# ---- Replace your HTTPStreamer class with this ----
class HTTPStreamer:
    """
    HTTP MJPEG streamer for live annotated frames.
    Uses a unique prefix to avoid route collisions and a default port 5050.
    """
    def __init__(self, port=5050, prefix="/_yolo_stream"):
        from flask import Flask
        import threading, time, socket, cv2

        self.app = Flask(__name__)
        self.port = port
        self.prefix = prefix.rstrip('/')
        self.running = False

        # Thread-safe cached JPEG bytes and a timestamp
        import threading
        self.lock = threading.Lock()
        self.latest_frame = None     # numpy BGR frame
        self.latest_jpeg = None      # bytes
        self.last_update = 0.0
        self._frame_count = 0

        self.setup_routes()

    def setup_routes(self):
        from flask import Response, make_response, send_file
        import numpy as np, cv2, io

        p = self.prefix

        @self.app.route(f'{p}/')
        def index():
            return f'''
            <!doctype html>
            <html>
            <head><title>YOLO Live Stream</title></head>
            <body style="background:#000;color:#fff;text-align:center">
              <h2>YOLO Live Detection</h2>
              <p>Snapshot: <a href="{p}/snapshot.jpg" target="_blank">{p}/snapshot.jpg</a></p>
              <img src="{p}/video_feed" style="max-width:95%;height:auto;border-radius:6px;border:1px solid #333"/>
              <p style="font-size:12px;color:#bbb">If the image is blank, open <code>{p}/snapshot.jpg</code> to test a single frame.</p>
            </body></html>
            '''

        @self.app.route(f'{p}/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route(f'{p}/snapshot.jpg')
        def snapshot():
            # return the latest jpeg bytes (single image) — useful for quick checks
            with self.lock:
                if self.latest_jpeg:
                    resp = make_response(self.latest_jpeg)
                    resp.headers['Content-Type'] = 'image/jpeg'
                    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                    return resp
            # if no frame yet, return a tiny black JPEG
            black = np.zeros((16,16,3), dtype='uint8')
            ret, buf = cv2.imencode('.jpg', black)
            resp = make_response(buf.tobytes())
            resp.headers['Content-Type'] = 'image/jpeg'
            return resp

    def generate_frames(self):
        import time
        # Continuous generator yielding the last encoded JPEG
        while self.running:
            with self.lock:
                data = self.latest_jpeg
            if data:
                try:
                    # each yield produces a multipart JPEG
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
                except GeneratorExit:
                    return
                except Exception as e:
                    print(f"[HTTPStreamer] yield error: {e}")
            else:
                # no frame yet — wait a bit
                time.sleep(0.05)
                continue
            # light throttle
            time.sleep(0.01)

    def add_frame(self, frame):
        """
        Call this from the predictor with a BGR numpy array (cv2 image).
        We encode immediately and cache bytes for faster serving.
        """
        import cv2, time
        if frame is None:
            return
        try:
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print("[HTTPStreamer] imencode failed")
                return
            jpg = buf.tobytes()
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_jpeg = jpg
                self.last_update = time.time()
                self._frame_count += 1
            # debug log (comment out if too verbose)
            if self._frame_count % 30 == 0:
                print(f"[HTTPStreamer] frames received: {self._frame_count}, last_update={self.last_update}")
        except Exception as e:
            print(f"[HTTPStreamer] add_frame error: {e}")

    def start_server(self):
        import threading, socket, time
        # Basic port check (non-blocking)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            res = sock.connect_ex(('127.0.0.1', self.port))
            sock.close()
            if res == 0:
                print(f"[HTTPStreamer] Warning: port {self.port} appears in use on 127.0.0.1")
        except Exception as e:
            print(f"[HTTPStreamer] port check error: {e}")

        def run_app():
            try:
                self.running = True
                # serve on all interfaces so other devices/OBS can connect
                self.app.run(host='0.0.0.0', port=self.port, debug=False,
                             use_reloader=False, threaded=True)
            except Exception as e:
                print(f"[HTTPStreamer] run_app error: {e}")
            finally:
                self.running = False

        t = threading.Thread(target=run_app, daemon=True)
        t.start()
        # wait a short time for the server to start
        time.sleep(0.8)
        prefix = self.prefix
        print(f"🌐 HTTP stream started at: http://localhost:{self.port}{prefix}/")
        print(f"   video feed -> http://localhost:{self.port}{prefix}/video_feed")
        print(f"   snapshot   -> http://localhost:{self.port}{prefix}/snapshot.jpg")
        return True
# ---- end HTTPStreamer replacement ----


# --- DetectionPredictor (unchanged internals except frame push to streamer) ---
class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       agnostic=self.args.agnostic_nms,
                                       max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        global fps_counter, fps_start_time, current_fps
        global http_streamer

        frame_start_time = time.time()

        p, im, im0 = batch
        all_outputs = []
        log_string = ""

        if len(im.shape) == 3:
            im = im[None]

        self.seen += 1
        im0 = im0.copy()

        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)

        if len(det) == 0:
            draw_boxes(im0, [], self.model.names, [], [])
            # push frame to HTTP streamer even when no detections so UI is always visible
            if ENABLE_HTTP_STREAM and http_streamer is not None:
                try:
                    http_streamer.add_frame(im0)
                except Exception:
                    pass
            return log_string

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # DeepSORT tracking
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []

        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))

        # If no boxes found, create empty tensors
        if len(xywh_bboxs) > 0:
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
        else:
            xywhs = torch.Tensor([])
            confss = torch.Tensor([])

        outputs = deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)
        else:
            draw_boxes(im0, [], self.model.names, [], [])

        # FPS and timing
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        frame_processing_times.append(frame_processing_time)

        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # --- PUSH annotated frame to HTTP streamer ---
        if ENABLE_HTTP_STREAM and http_streamer is not None:
            try:
                http_streamer.add_frame(im0)
            except Exception:
                print("HTTPStreamer.add_frame error:", e)

        return log_string

# --- predict entrypoint (start http_streamer here) ---
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    global http_streamer
    start_logging_services()
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)

    # start HTTP streamer if enabled
    if ENABLE_HTTP_STREAM:
        try:
            http_streamer = HTTPStreamer(port=5050, prefix="/_yolo_stream")
            started = http_streamer.start_server()
            if not started:
                http_streamer = None
        except Exception as e:
            print(f"[Main] Failed to start HTTP streamer: {e}")
            http_streamer = None

    # Enhanced source handling for RTSP
    if cfg.source is None:
        cfg.source = ROOT / "assets"
    elif isinstance(cfg.source, str) and cfg.source.startswith('rtsp://'):
        print(f"RTSP source detected: {cfg.source}")

    predictor = DetectionPredictor(cfg)
    # run predictor (unchanged)
    predictor()

if __name__ == "__main__":
    predict()
