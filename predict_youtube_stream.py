#!/usr/bin/env python3
"""
YouTube Live streaming prediction script with Line Crossing Counting.

Features:
- Real-time Object Detection & Tracking (YOLO + DeepSORT)
- Line Crossing Counting (People, Motor, Car)
- YouTube Live Streaming via FFmpeg/RTMP
- Auto-restart before YouTube's 12-hour limit (for 24/7 streaming)
- Speed Violation Detection & Plate Capture
- Counter API Integration for persistence
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time
import threading
import datetime
import requests

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import (
    InferenceConfig, TrackingConfig,
    VisualizationConfig, YouTubeStreamingConfig, PlateCaptureConfig
)
from src.streaming.youtube_streamer import YouTubeStreamer
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

    def __init__(self, api_url, camera_id, interval=10, api_key=None):
        self.api_url = api_url.rstrip("/")
        self.camera_id = camera_id
        self.interval = interval
        self.api_key = api_key
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
                headers = {}
                if self.api_key:
                    headers["X-API-Key"] = self.api_key
                resp = requests.post(
                    f"{self.api_url}/api/ingest",
                    json=payload,
                    headers=headers,
                    timeout=5,
                )
                print(f"[Ingest] POST {resp.status_code}: {counts}")
            except Exception as e:
                print(f"[Ingest] POST failed: {e}")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3)


class StreamRegistrationClient:
    """Client for registering/unregistering streams with external API.

    Uses Basic Auth and POSTs stream info when starting,
    DELETEs when stopping or restarting. This allows a frontend
    to dynamically list active streams via GET /api/streaming/list.
    """

    def __init__(self, api_url, username, password):
        self.api_url = api_url.rstrip("/")
        self.username = username
        self.password = password
        self._current_stream_id = None

    def _get_auth_header(self):
        """Build Basic Auth header."""
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def register_stream(self, stream_id, title="", description="", stream_type="", location=""):
        """POST to register a new stream."""
        try:
            payload = {
                "streamingId": stream_id,
                "title": title,
                "description": description,
            }
            if stream_type:
                payload["type"] = stream_type
            if location:
                payload["location"] = location
            resp = requests.post(
                f"{self.api_url}/api/streaming/",
                json=payload,
                headers=self._get_auth_header(),
                timeout=10,
            )
            if resp.status_code in (200, 201):
                self._current_stream_id = stream_id
                print(f"[Registration] Registered stream: {stream_id}")
            else:
                print(f"[Registration] Failed to register stream: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[Registration] Error registering stream: {e}")

    def unregister_stream(self):
        """DELETE to unregister current stream."""
        if not self._current_stream_id:
            return

        try:
            resp = requests.delete(
                f"{self.api_url}/api/streaming/{self._current_stream_id}",
                headers=self._get_auth_header(),
                timeout=10,
            )
            if resp.status_code in (200, 204):
                print(f"[Registration] Unregistered stream: {self._current_stream_id}")
            else:
                print(f"[Registration] Failed to unregister stream: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[Registration] Error unregistering stream: {e}")
        finally:
            self._current_stream_id = None


class StreamRestartManager:
    """
    Manages automatic YouTube stream restarts to handle YouTube's 12-hour limit.
    Monitors stream duration and seamlessly restarts before hitting the limit.
    """

    # YouTube has a 12-hour limit; restart at 11h 55m to be safe
    MAX_STREAM_DURATION_SECONDS = 11 * 60 * 60 + 55 * 60  # 11 hours 55 minutes

    def __init__(self, youtube_config: YouTubeStreamingConfig, registration_client: StreamRegistrationClient = None,
                 stream_type: str = "", location: str = ""):
        self.youtube_config = youtube_config
        self.registration_client = registration_client
        self.stream_type = stream_type
        self.location = location
        self.streamer: YouTubeStreamer = None
        self.lock = threading.Lock()
        self.stream_start_time: float = None
        self.restart_count = 0
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread = None

    def start(self) -> bool:
        """Start the YouTube streamer and monitoring thread."""
        with self.lock:
            self.streamer = YouTubeStreamer(self.youtube_config)
            if not self.streamer.start():
                return False
            self.stream_start_time = time.time()
            self.restart_count = 0

            # Register stream with external API
            if self.registration_client and self.streamer.broadcast_id:
                self.registration_client.register_stream(
                    stream_id=self.streamer.broadcast_id,
                    title=self.youtube_config.broadcast_title,
                    description=self.youtube_config.broadcast_description,
                    stream_type=self.stream_type,
                    location=self.location,
                )

        # Start monitoring thread
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        return True

    def _monitor_loop(self):
        """Background thread that checks stream duration and triggers restart."""
        while not self._stop_event.wait(30):  # Check every 30 seconds
            with self.lock:
                if self.streamer is None or self.stream_start_time is None:
                    continue

                elapsed = time.time() - self.stream_start_time
                remaining = self.MAX_STREAM_DURATION_SECONDS - elapsed

                # Log every 30 minutes
                if int(elapsed) % 1800 < 30:
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    print(f"[StreamManager] Stream duration: {hours}h {minutes}m")

                if remaining <= 0:
                    print(f"[StreamManager] Approaching 12-hour limit, restarting stream...")
                    self._restart_stream_locked()

    def _restart_stream_locked(self):
        """Restart the stream. Must be called with lock held."""
        # Unregister current stream before stopping
        if self.registration_client:
            self.registration_client.unregister_stream()

        # Stop current stream
        if self.streamer:
            try:
                self.streamer.stop()
            except Exception as e:
                print(f"[StreamManager] Error stopping stream: {e}")

        # Brief pause to let YouTube process the end
        time.sleep(3)

        # Start new stream
        self.streamer = YouTubeStreamer(self.youtube_config)
        if self.streamer.start():
            self.stream_start_time = time.time()
            self.restart_count += 1
            print(f"[StreamManager] Stream restarted successfully (restart #{self.restart_count})")

            # Register new stream with external API
            if self.registration_client and self.streamer.broadcast_id:
                self.registration_client.register_stream(
                    stream_id=self.streamer.broadcast_id,
                    title=self.youtube_config.broadcast_title,
                    description=self.youtube_config.broadcast_description,
                    stream_type=self.stream_type,
                    location=self.location,
                )

            # Wait for FFmpeg to be ready
            time.sleep(5)
            if not self.streamer.check_ffmpeg_health():
                print("[StreamManager] Warning: FFmpeg health check failed after restart")
        else:
            print("[StreamManager] Failed to restart stream!")

    def add_frame(self, frame: np.ndarray):
        """Thread-safe frame addition to current streamer."""
        with self.lock:
            if self.streamer and self.streamer.is_running():
                self.streamer.add_frame(frame)

    def is_running(self) -> bool:
        """Check if streamer is running."""
        with self.lock:
            return self.streamer is not None and self.streamer.is_running()

    def check_ffmpeg_health(self) -> bool:
        """Check FFmpeg health."""
        with self.lock:
            if self.streamer:
                return self.streamer.check_ffmpeg_health()
            return False

    def get_stats(self) -> dict:
        """Get stats from current streamer."""
        with self.lock:
            if self.streamer:
                stats = self.streamer.get_stats()
                stats['restart_count'] = self.restart_count
                if self.stream_start_time:
                    stats['current_stream_duration'] = time.time() - self.stream_start_time
                return stats
            return {'restart_count': self.restart_count}

    def stop(self):
        """Stop the streamer and monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        with self.lock:
            # Unregister stream before final shutdown
            if self.registration_client:
                self.registration_client.unregister_stream()

            if self.streamer:
                self.streamer.stop()
                self.streamer = None


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
                                print(f"[Counter] Object crossed line: {key} (ID: {track_id})")

                # Update history
                self.previous_centroids[track_id] = current_center

            # Clean up old tracks
            for tid in list(self.previous_centroids.keys()):
                if tid not in current_ids:
                    del self.previous_centroids[tid]


class YouTubeStreamInference(InferenceEngine):
    """Extended inference engine with YouTube Live streaming, counting, and plate capture."""

    def __init__(
        self,
        stream_manager: StreamRestartManager,
        traffic_counter: TrafficCounter,
        plate_capture_manager: PlateCaptureManager = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stream_manager = stream_manager
        self.traffic_counter = traffic_counter
        self.plate_capture_manager = plate_capture_manager

    def write_results(self, idx: int, preds, batch):
        """Override to run counting logic and YouTube streaming."""
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

            # 3. Push to YouTube Streamer (via StreamRestartManager)
            if self.stream_manager and self.stream_manager.is_running():
                try:
                    self.stream_manager.add_frame(self.annotated_frame)
                except Exception as e:
                    print(f"Error pushing frame to YouTube stream: {e}")

            # 4. Check for Speed Violations
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
                        print(f"[Capture] Violation: Track {track_id} @ {speed:.1f} km/h")
                except Exception:
                    continue
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection with YouTube Live Streaming'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Video file path or RTSP stream URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS (default: 0.7)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on: cuda or cpu (default: cuda)')
    parser.add_argument('--show', action='store_true',
                       help='Also display results in window')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Inference image size (default: 640)')
    parser.add_argument('--counter-direction', type=str, default='north_enter',
                       choices=['north_enter', 'south_enter'],
                       help='Counter direction: north_enter (North=Enter, South=Exit) or south_enter (South=Enter, North=Exit) (default: north_enter)')
    parser.add_argument('--show-info-panel', action='store_true', default=True,
                       help='Show system information panel (default: True)')
    parser.add_argument('--hide-info-panel', action='store_true',
                       help='Hide system information panel')
    parser.add_argument('--show-counters', action='store_true', default=True,
                       help='Show object counters (default: True)')
    parser.add_argument('--hide-counters', action='store_true',
                       help='Hide object counters')
    parser.add_argument('--enable-keyboard-reset', action='store_true', default=True,
                       help='Enable R key to reset counters (default: True)')
    parser.add_argument('--disable-keyboard-reset', action='store_true',
                       help='Disable R key reset functionality')
    parser.add_argument('--enable-auto-reset', action='store_true', default=True,
                       help='Enable automatic daily reset (default: True)')
    parser.add_argument('--disable-auto-reset', action='store_true',
                       help='Disable automatic daily reset')
    
    # YouTube streaming specific arguments
    parser.add_argument('--youtube-title', type=str, 
                       default='CCTV Teknik Test Stream',
                       help='YouTube broadcast title')
    parser.add_argument('--youtube-description', type=str,
                       default='Real-time object detection and tracking using YOLO',
                       help='YouTube broadcast description')
    parser.add_argument('--youtube-privacy', type=str, default='unlisted',
                       choices=['public', 'private', 'unlisted'],
                       help='YouTube broadcast privacy (default: public)')
    parser.add_argument('--youtube-resolution', type=str, default='720p',
                       choices=['480p', '720p', '1080p'],
                       help='YouTube stream resolution (default: 720p)')
    parser.add_argument('--youtube-bitrate', type=int, default=2500,
                       help='YouTube stream video bitrate in kbps (default: 2500)')
    parser.add_argument('--client-secrets', type=str, default='client_secrets.json',
                       help='Path to YouTube API client secrets file')
    parser.add_argument('--manual-transition', action='store_true',
                       help='Manually transition to live (instead of automatic)')

    # Plate capture arguments
    parser.add_argument('--enable-plate-capture', action='store_true', default=True,
                       help='Enable speed violation capture (default: True)')
    parser.add_argument('--disable-plate-capture', action='store_true',
                       help='Disable speed violation capture')
    parser.add_argument('--speed-limit', type=float, default=60.0,
                       help='Speed limit in km/h for violation detection (default: 60.0)')
    parser.add_argument('--violation-output-dir', type=str, default='output/violations',
                       help='Directory to save violation captures (default: output/violations)')
    parser.add_argument('--capture-quality', type=int, default=95,
                       help='JPEG quality for captured images (default: 95)')
    parser.add_argument('--pixels-per-meter', type=float, default=20.0,
                       help='Pixels per meter for speed calculation (default: 20.0)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Video FPS for speed calculation (default: 30.0)')

    # Counter persistence arguments
    parser.add_argument('--counter-api-url', type=str, default=None,
                       help='URL of Hono counter service (e.g. http://localhost:3000). If unset, persistence disabled.')
    parser.add_argument('--camera-id', type=str, default='default',
                       help='Unique camera identifier (e.g. "jalan-masuk-utama")')
    parser.add_argument('--ingest-interval', type=int, default=10,
                       help='Seconds between POSTs to counter API (default: 10)')
    parser.add_argument('--counter-api-key', type=str, default=None,
                       help='API key for authenticating with the counter service')

    # Stream registration arguments
    parser.add_argument('--registration-api-url', type=str, default=None,
                       help='URL of stream registration service (e.g. http://localhost:3000). If unset, registration disabled.')
    parser.add_argument('--registration-username', type=str, default=None,
                       help='Basic auth username for registration API')
    parser.add_argument('--registration-password', type=str, default=None,
                       help='Basic auth password for registration API')
    parser.add_argument('--location', type=str, default=None,
                       help='Location identifier for grouping streams (e.g. "jalan-masuk-utama")')

    # Logging control
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging (frame info, track updates, etc.)')

    args = parser.parse_args()
    
    # Handle visibility and feature logic
    show_info_panel = args.show_info_panel and not args.hide_info_panel
    show_counters = args.show_counters and not args.hide_counters
    enable_keyboard_reset = args.enable_keyboard_reset and not args.disable_keyboard_reset
    enable_auto_reset = args.enable_auto_reset and not args.disable_auto_reset
    enable_plate_capture = args.enable_plate_capture and not args.disable_plate_capture

    # Initialize Traffic Counter
    counter = TrafficCounter(COUNTING_LINE_START, COUNTING_LINE_END)

    # Initialize Counter Ingest Client (optional persistence)
    ingest_client = None
    if args.counter_api_url:
        ingest_client = CounterIngestClient(
            api_url=args.counter_api_url,
            camera_id=args.camera_id,
            interval=args.ingest_interval,
            api_key=args.counter_api_key,
        )
        print(f"   Counter API: {args.counter_api_url} (camera: {args.camera_id}, every {args.ingest_interval}s)")

    # Initialize Stream Registration Client (optional)
    registration_client = None
    if args.registration_api_url:
        if not args.registration_username or not args.registration_password:
            print("Warning: --registration-api-url provided but missing --registration-username or --registration-password. Registration disabled.")
        else:
            registration_client = StreamRegistrationClient(
                api_url=args.registration_api_url,
                username=args.registration_username,
                password=args.registration_password,
            )
            print(f"   Registration API: {args.registration_api_url}")

    # Create configurations
    inference_config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        image_size=args.imgsz
    )
    
    tracking_config = TrackingConfig(verbose=args.verbose)
    visualization_config = VisualizationConfig(
        counter_direction=args.counter_direction,
        show_info_panel=show_info_panel,
        show_counters=show_counters,
        enable_keyboard_reset=enable_keyboard_reset,
        enable_auto_daily_reset=enable_auto_reset
    )
    
    youtube_config = YouTubeStreamingConfig(
        enabled=True,
        client_secrets_file=args.client_secrets,
        broadcast_title=args.youtube_title,
        broadcast_description=args.youtube_description,
        privacy_status=args.youtube_privacy,
        resolution=args.youtube_resolution,
        video_bitrate=args.youtube_bitrate,
        auto_start=not args.manual_transition  # Disable auto-start if manual transition requested
    )

    plate_capture_config = PlateCaptureConfig(
        enabled=enable_plate_capture,
        output_dir=args.violation_output_dir,
        speed_limit=args.speed_limit,
        image_quality=args.capture_quality
    )

    # Hydra config for BasePredictor
    # Ensure imgsz is a list for compatibility
    imgsz = [args.imgsz, args.imgsz] if isinstance(args.imgsz, int) else args.imgsz
    
    hydra_config = {
        'model': args.model,
        'source': args.source,
        'conf': args.conf,
        'iou': args.iou,
        'device': args.device,
        'imgsz': imgsz,
        'show': args.show,
        'save': args.save,
        'verbose': args.verbose,  # Suppress per-frame logs unless --verbose is set
    }
    
    print("Starting YOLO YouTube Live Streaming...")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Confidence: {args.conf}")
    print(f"   YouTube Title: {args.youtube_title}")
    print(f"   YouTube Privacy: {args.youtube_privacy}")
    print(f"   YouTube Resolution: {args.youtube_resolution}")
    if args.manual_transition:
        print(f"   Transition Mode: Manual")
    if enable_plate_capture:
        print(f"   Plate Capture: Enabled (speed limit: {args.speed_limit} km/h)")
    print()

    # Initialize Plate Capture Manager
    plate_capture_manager = PlateCaptureManager(
        output_dir=plate_capture_config.output_dir,
        speed_limit=plate_capture_config.speed_limit,
        enabled=plate_capture_config.enabled,
        save_metadata=plate_capture_config.save_metadata,
        image_format=plate_capture_config.image_format,
        image_quality=plate_capture_config.image_quality
    )

    # Initialize Stream Manager (handles auto-restart for 12-hour YouTube limit)
    stream_manager = StreamRestartManager(
        youtube_config,
        registration_client=registration_client,
        stream_type="with-inference",
        location=args.location or "",
    )
    print("   Auto-restart: Enabled (restarts before 12-hour YouTube limit)")
    if registration_client:
        print("   Stream Registration: Enabled")

    # Start YouTube stream via manager
    if not stream_manager.start():
        print("Failed to start YouTube stream. Exiting.")
        return 1

    # Wait for stream to initialize and FFmpeg to be ready
    print("Waiting for YouTube stream to initialize...")
    time.sleep(5)

    # Verify FFmpeg is healthy before starting inference
    if not stream_manager.check_ffmpeg_health():
        print("FFmpeg process is not healthy. Exiting.")
        return 1

    print("YouTube stream initialized and ready for frames")
    
    # Initialize inference engine with YouTube streaming
    engine = YouTubeStreamInference(
        stream_manager=stream_manager,
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
        print("🚀 Starting inference and streaming to YouTube...")
        print("   Press Ctrl+C to stop streaming")
        if args.manual_transition:
            print("   Note: Use manual transition to go live when ready")
        else:
            print("   Note: Stream will go live automatically when data is received")
        print()
        
        # Run prediction
        results = engine()
        print("\n✅ Inference completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
    finally:
        # Stop ingest client
        if ingest_client:
            ingest_client.stop()

        # Always stop the stream manager
        stream_manager.stop()

        # Print plate capture statistics
        if enable_plate_capture:
            plate_capture_manager.print_statistics()

        # Print final statistics
        stats = stream_manager.get_stats()
        print(f"\nFinal Stream Statistics:")
        print(f"   Total frames streamed: {stats.get('frame_count', 0)}")
        print(f"   Total stream restarts: {stats.get('restart_count', 0)}")
        if stats.get('duration'):
            print(f"   Last stream duration: {stats['duration']:.1f} seconds")
            print(f"   Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"   Stream URL: {stats['stream_url']}")


if __name__ == "__main__":
    main()