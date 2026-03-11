#!/usr/bin/env python3
"""
Refactored Live Streaming Prediction with Line Crossing Counting and REST API.

Features:
- Real-time Object Detection & Tracking (YOLO + DeepSORT)
- Line Crossing Counting (People, Motor, Car)
- HTTP/MJPEG Streaming (Video Feed)
- REST API (/data) for real-time analytics (Server-Sent Events)
- Thread-safe implementations
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time
import threading
import datetime
import json
import requests
from flask import jsonify, Response, stream_with_context

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import (
    InferenceConfig, TrackingConfig, 
    VisualizationConfig, StreamingConfig, PlateCaptureConfig
)
from src.streaming.server import StreamingServer
from src.capture.plate_capture import PlateCaptureManager

# =================================================================================================
#  USER CONFIGURATION SECTION
#  Adjust these coordinates for the counting line.
#  Format: (x, y)
#  (0,0) is Top-Left.
# =================================================================================================
COUNTING_LINE_START = (100, 420)    # Start point of the line (Left)
COUNTING_LINE_END = (1050, 420)    # End point of the line (Right)
# =================================================================================================

class CounterIngestClient:
    """Daemon thread that periodically POSTs absolute counts to the Hono counter API.

    Reads counts from the renderer's actual counters (object_counter_in / object_counter_out)
    which are the source of truth for line-crossing detection.
    """

    def __init__(self, api_url, camera_id, interval=10):
        self.api_url = api_url.rstrip("/")
        self.camera_id = camera_id
        self.interval = interval
        self.renderer = None  # Set after engine is created
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_renderer(self, renderer):
        """Attach the renderer whose counters we read from."""
        self.renderer = renderer

    def _get_counts(self):
        """Build counts dict from the renderer's actual counters."""
        r = self.renderer
        if r is None:
            return None
        # renderer uses: object_counter_in["person"], object_counter_out["car"], etc.
        # Map to the API format: people_in, people_out, motor_in, motor_out, car_in, car_out
        name_map = {"person": "people", "motorcycle": "motor", "car": "car"}
        counts = {}
        for orig, mapped in name_map.items():
            counts[f"{mapped}_in"] = r.object_counter_in.get(orig, 0)
            counts[f"{mapped}_out"] = r.object_counter_out.get(orig, 0)
        return counts

    def _run(self):
        while not self._stop_event.wait(self.interval):
            try:
                counts = self._get_counts()
                if counts is None:
                    continue
                payload = {
                    "camera_id": self.camera_id,
                    "counts": counts,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                resp = requests.post(
                    f"{self.api_url}/api/ingest",
                    json=payload,
                    timeout=5,
                )
                print(f"📤 Ingest POST {resp.status_code}: {counts}")
            except Exception as e:
                print(f"⚠️  Ingest POST failed: {e}")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3)


class TrafficCounter:
    """
    Thread-safe counter for line crossing logic.
    Maintains history of object positions to detect line crossing.
    """
    def __init__(self, line_start, line_end):
        self.lock = threading.Lock()
        self.line_start = np.array(line_start)
        self.line_end = np.array(line_end)
        
        # Counts
        self.counts = {
            "people_in": 0, "people_out": 0,
            "motor_in": 0, "motor_out": 0,
            "car_in": 0, "car_out": 0
        }
        
        # Mapping Class ID to Name
        # 0: person, 2: car, 3: motorcycle 
        # (Based on standard COCO, adjust if model differs)
        self.class_mapping = {
            0: "people",
            2: "car",
            3: "motor"
        }
        
        # History: track_id -> previous_center (numpy array)
        self.previous_centroids = {}

    def get_counts(self):
        """Return a copy of the current counts safe for API response."""
        with self.lock:
            data = self.counts.copy()
            data['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return data

    def _ccw(self, A, B, C):
        """Check counter-clockwise order of points A, B, C."""
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def _intersect(self, A, B, C, D):
        """Return true if line segments AB and CD intersect."""
        return self._ccw(A,C,D) != self._ccw(B,C,D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def update(self, tracks):
        """
        Update counts based on tracked objects.
        tracks: Dictionary of TrackedObject from ObjectTracker
        """
        current_ids = set()
        
        with self.lock:
            for track_id, track in tracks.items():
                current_ids.add(track_id)
                
                # Get current centroid
                bbox = track.bbox
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                current_center = np.array([cx, cy])
                
                # Check if we have history for this object
                if track_id in self.previous_centroids:
                    prev_center = self.previous_centroids[track_id]
                    
                    # Check intersection with counting line
                    if self._intersect(prev_center, current_center, self.line_start, self.line_end):
                        
                        dy = current_center[1] - prev_center[1]
                        
                        direction = "in" if dy > 0 else "out" # Simple Y-axis logic
                        
                        class_key = self.class_mapping.get(track.class_id)
                        
                        if class_key:
                            key = f"{class_key}_{direction}"
                            if key in self.counts:
                                self.counts[key] += 1
                                print(f"📍 object crossed line: {key} (ID: {track_id})")
                
                # Update history
                self.previous_centroids[track_id] = current_center
            
            # Clean up old tracks
            for tid in list(self.previous_centroids.keys()):
                if tid not in current_ids:
                    del self.previous_centroids[tid]

class LiveStreamInference(InferenceEngine):
    """Extended inference engine with HTTP streaming, counting, and plate capture."""
    
    def __init__(
        self,
        streaming_server: StreamingServer,
        traffic_counter: TrafficCounter,
        plate_capture_manager: PlateCaptureManager = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.streaming_server = streaming_server
        self.plate_capture_manager = plate_capture_manager
        self.traffic_counter = traffic_counter
        
    def write_results(self, idx: int, preds, batch):
        """Override to run counting logic and streaming."""
        log_string = super().write_results(idx, preds, batch)
        
        # 1. Update Tracking Counts
        if hasattr(self, 'tracker') and self.tracker.tracks:
            self.traffic_counter.update(self.tracker.tracks)

        # 2. Draw Counting Line & Annotations on the Frame
        if hasattr(self, 'annotated_frame') and self.annotated_frame is not None:
            import cv2
            
            # Draw line
            cv2.line(
                self.annotated_frame, 
                tuple(COUNTING_LINE_START), 
                tuple(COUNTING_LINE_END), 
                (0, 255, 255), 2
            )
            
            # Draw Counts (Simple Overlay)
            counts = self.traffic_counter.counts
            text = f"P In: {counts['people_in']} Out: {counts['people_out']} | C In: {counts['car_in']} Out: {counts['car_out']}"
            cv2.putText(self.annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 3. Push to Streaming Server
            if self.streaming_server and self.streaming_server.running:
                try:
                    self.streaming_server.add_frame(self.annotated_frame)
                except Exception as e:
                    print(f"Error pushing frame to stream: {e}")

            # 4. Check for Speed Violations (Existing Logic)
            if self.plate_capture_manager and self.plate_capture_manager.enabled:
                self._check_speed_violations(batch)
                
        return log_string

    def _check_speed_violations(self, batch):
        """Check tracked objects for speed violations and capture."""
        if not self.plate_capture_manager or not self.plate_capture_manager.enabled:
            return
        if not hasattr(self, 'tracker') or self.tracker is None:
            return
        try:
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                im0s = batch[2]
            else:
                return
            frame = im0s[0] if isinstance(im0s, list) else im0s
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return
            if not hasattr(self.tracker, 'tracks') or len(self.tracker.tracks) == 0:
                return
            
            for track_id, track in self.tracker.tracks.items():
                try:
                    speed = track.speed
                    if speed is None or speed <= 0:
                        continue
                    if not track.is_confirmed():
                        continue
                    bbox = track.to_tlbr()
                    class_name = f"class_{track.class_id}"
                    
                    captured_path = self.plate_capture_manager.check_and_capture(
                        frame=frame,
                        track_id=track_id,
                        bbox=tuple(map(int, bbox)),
                        speed=speed,
                        class_name=class_name
                    )
                    if captured_path:
                        print(f"✅ Captured violation: Track {track_id} @ {speed:.1f} km/h")
                except Exception:
                    continue
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description='Refactored YOLO Live Stream with Counters & API')
    
    # Standard Arguments
    parser.add_argument('--source', type=str, required=True, help='Video file path or RTSP stream URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--port', type=int, default=5050, help='HTTP streaming port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='HTTP server host')
    parser.add_argument('--show', action='store_true', help='Also display results in window')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    # Existing features arguments
    parser.add_argument('--counter-direction', type=str, default='north_enter', choices=['north_enter', 'south_enter'])
    parser.add_argument('--show-info-panel', action='store_true', default=True)
    parser.add_argument('--hide-info-panel', action='store_true')
    parser.add_argument('--show-counters', action='store_true', default=True)
    parser.add_argument('--hide-counters', action='store_true')
    parser.add_argument('--enable-keyboard-reset', action='store_true', default=True)
    parser.add_argument('--disable-keyboard-reset', action='store_true')
    parser.add_argument('--enable-auto-reset', action='store_true', default=True)
    parser.add_argument('--disable-auto-reset', action='store_true')
    
    # Plate capture arguments
    parser.add_argument('--enable-plate-capture', action='store_true', default=True)
    parser.add_argument('--disable-plate-capture', action='store_true')
    parser.add_argument('--speed-limit', type=float, default=60.0)
    parser.add_argument('--violation-output-dir', type=str, default='output/violations')
    parser.add_argument('--capture-quality', type=int, default=95)
    parser.add_argument('--pixels-per-meter', type=float, default=20.0)
    parser.add_argument('--fps', type=float, default=30.0)

    # Counter persistence arguments
    parser.add_argument('--counter-api-url', type=str, default=None, help='URL of Hono counter service (e.g. http://localhost:3000). If unset, persistence disabled.')
    parser.add_argument('--camera-id', type=str, default='default', help='Unique camera identifier (e.g. "jalan-masuk-utama")')
    parser.add_argument('--ingest-interval', type=int, default=10, help='Seconds between POSTs to counter API')

    args = parser.parse_args()

    # Initialize Logic
    # 1. Traffic Counter
    counter = TrafficCounter(COUNTING_LINE_START, COUNTING_LINE_END)

    # 1b. Counter Ingest Client (optional persistence)
    ingest_client = None
    if args.counter_api_url:
        ingest_client = CounterIngestClient(
            api_url=args.counter_api_url,
            camera_id=args.camera_id,
            interval=args.ingest_interval,
        )
        print(f"   Counter API: {args.counter_api_url} (camera: {args.camera_id}, every {args.ingest_interval}s)")

    # 2. Configs
    show_info_panel = args.show_info_panel and not args.hide_info_panel
    show_counters = args.show_counters and not args.hide_counters
    enable_keyboard_reset = args.enable_keyboard_reset and not args.disable_keyboard_reset
    enable_auto_reset = args.enable_auto_reset and not args.disable_auto_reset
    enable_plate_capture = args.enable_plate_capture and not args.disable_plate_capture

    inference_config = InferenceConfig(
        model_path=args.model, confidence_threshold=args.conf,
        iou_threshold=args.iou, device=args.device,
        image_size=[640, 640]
    )
    tracking_config = TrackingConfig()
    visualization_config = VisualizationConfig(
        counter_direction=args.counter_direction, show_info_panel=show_info_panel,
        show_counters=show_counters, enable_keyboard_reset=enable_keyboard_reset,
        enable_auto_daily_reset=enable_auto_reset
    )
    streaming_config = StreamingConfig(enabled=True, host=args.host, port=args.port)
    plate_capture_config = PlateCaptureConfig(
        enabled=enable_plate_capture, output_dir=args.violation_output_dir,
        speed_limit=args.speed_limit, image_quality=args.capture_quality
    )

    hydra_config = {
        'model': args.model, 'source': args.source, 'conf': args.conf,
        'iou': args.iou, 'device': args.device, 'imgsz': [640, 640],
        'show': args.show, 'save': args.save,
    }

    # 3. Streaming Server & API
    print(f"🚀 Starting Refactored YOLO System...")
    print(f"   API Endpoint: http://{args.host}:{args.port}/data")
    print(f"   Video Stream: http://{args.host}:{args.port}/_yolo_stream/video_feed")

    streaming_server = StreamingServer(streaming_config)
    
    # Inject API Route with SSE
    @streaming_server.app.route('/data')
    def api_data():
        def generate():
            last_counts = None
            last_sent_time = 0
            
            while True:
                with counter.lock:
                    current_counts = counter.counts.copy()
                
                current_time = time.time()
                
                # Condition 1: Data Changed (Immediate Event)
                data_changed = current_counts != last_counts
                
                # Condition 2: Heartbeat (Every 1.0s)
                time_since_last = current_time - last_sent_time
                heartbeat_due = time_since_last >= 1.0
                
                if data_changed or heartbeat_due:
                    # Prepare payload
                    payload = current_counts.copy()
                    payload['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    json_data = json.dumps(payload)
                    yield f"data: {json_data}\n\n"
                    
                    # Update states
                    last_counts = current_counts
                    last_sent_time = current_time
                
                # Check frequent enough to be responsive (20Hz), but sleep to save CPU
                time.sleep(0.05)
        
        response = Response(stream_with_context(generate()), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    @streaming_server.app.route('/stats')
    def api_stats():
        """Snapshot endpoint for non-streaming clients (e.g. Postman)"""
        with counter.lock:
            payload = counter.counts.copy()
        payload['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(payload)

    streaming_server.start()
    time.sleep(1)

    # 4. Plate Capture Manager
    plate_capture_manager = PlateCaptureManager(
        output_dir=plate_capture_config.output_dir,
        speed_limit=plate_capture_config.speed_limit,
        enabled=plate_capture_config.enabled,
        save_metadata=plate_capture_config.save_metadata,
        image_format=plate_capture_config.image_format,
        image_quality=plate_capture_config.image_quality
    )

    # 5. Inference Engine
    engine = LiveStreamInference(
        streaming_server=streaming_server,
        traffic_counter=counter,
        plate_capture_manager=plate_capture_manager,
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )

    # Connect ingest client to the renderer's actual counters
    if ingest_client:
        ingest_client.set_renderer(engine.renderer)

    try:
        results = engine()
        print("\n✅ Inference completed!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        if ingest_client:
            ingest_client.stop()
        streaming_server.stop()
        if enable_plate_capture:
            plate_capture_manager.print_statistics()

if __name__ == "__main__":
    main()
