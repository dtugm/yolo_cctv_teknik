#!/usr/bin/env python3
"""
Test script for YouTube stream auto-restart functionality.

This script tests the StreamRestartManager by restarting the stream every 5 minutes
instead of the production 11h 55m limit. Use this to verify the restart logic works
before deploying for 24/7 streaming.

Usage:
    python test_youtube_restream.py --source <video_or_rtsp_url>
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time
import threading
import requests
import base64

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import (
    InferenceConfig, TrackingConfig,
    VisualizationConfig, YouTubeStreamingConfig
)
from src.streaming.youtube_streamer import YouTubeStreamer

# =================================================================================================
#  TEST CONFIGURATION
# =================================================================================================
TEST_RESTART_INTERVAL_SECONDS = 5 * 60  # 5 minutes for testing
COUNTING_LINE_START = (100, 420)
COUNTING_LINE_END = (1050, 420)
# =================================================================================================


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
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}

    def register_stream(self, stream_id, title="", description=""):
        """POST to register a new stream."""
        try:
            payload = {
                "streamingId": stream_id,
                "title": title,
                "description": description,
            }
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


class TestStreamRestartManager:
    """
    Test version of StreamRestartManager with 5-minute restart interval.
    Logs more frequently to help verify the restart logic.
    """

    def __init__(self, youtube_config: YouTubeStreamingConfig, restart_interval: int = TEST_RESTART_INTERVAL_SECONDS, registration_client: StreamRegistrationClient = None):
        self.youtube_config = youtube_config
        self.restart_interval = restart_interval
        self.registration_client = registration_client
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
                )

        # Start monitoring thread
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        print(f"[TEST] Stream started. Will restart in {self.restart_interval} seconds ({self.restart_interval // 60} minutes)")
        return True

    def _monitor_loop(self):
        """Background thread that checks stream duration and triggers restart."""
        while not self._stop_event.wait(10):  # Check every 10 seconds for faster feedback
            with self.lock:
                if self.streamer is None or self.stream_start_time is None:
                    continue

                elapsed = time.time() - self.stream_start_time
                remaining = self.restart_interval - elapsed

                # Log every 30 seconds for testing
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                remaining_minutes = int(remaining // 60)
                remaining_seconds = int(remaining % 60)
                print(f"[TEST] Stream duration: {minutes}m {seconds}s | Restart in: {remaining_minutes}m {remaining_seconds}s")

                if remaining <= 0:
                    print(f"[TEST] Restart interval reached! Restarting stream...")
                    self._restart_stream_locked()

    def _restart_stream_locked(self):
        """Restart the stream. Must be called with lock held."""
        print("[TEST] Stopping current stream...")

        # Unregister old stream before stopping
        if self.registration_client:
            self.registration_client.unregister_stream()

        # Stop current stream
        if self.streamer:
            try:
                self.streamer.stop()
            except Exception as e:
                print(f"[TEST] Error stopping stream: {e}")

        # Brief pause to let YouTube process the end
        print("[TEST] Waiting 3 seconds before starting new stream...")
        time.sleep(3)

        # Start new stream
        print("[TEST] Starting new stream...")
        self.streamer = YouTubeStreamer(self.youtube_config)
        if self.streamer.start():
            self.stream_start_time = time.time()
            self.restart_count += 1
            print(f"[TEST] Stream restarted successfully! (restart #{self.restart_count})")

            # Register new stream
            if self.registration_client and self.streamer.broadcast_id:
                self.registration_client.register_stream(
                    stream_id=self.streamer.broadcast_id,
                    title=self.youtube_config.broadcast_title,
                    description=self.youtube_config.broadcast_description,
                )

            # Wait for FFmpeg to be ready
            print("[TEST] Waiting for FFmpeg to initialize...")
            time.sleep(5)
            if self.streamer.check_ffmpeg_health():
                print("[TEST] FFmpeg is healthy, resuming frame streaming")
            else:
                print("[TEST] Warning: FFmpeg health check failed after restart")
        else:
            print("[TEST] Failed to restart stream!")

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
            # Unregister stream before stopping
            if self.registration_client:
                self.registration_client.unregister_stream()

            if self.streamer:
                self.streamer.stop()
                self.streamer = None


class TestYouTubeStreamInference(InferenceEngine):
    """Simplified inference engine for testing YouTube streaming."""

    def __init__(
        self,
        stream_manager: TestStreamRestartManager,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stream_manager = stream_manager
        self._frame_count = 0

    def write_results(self, idx: int, preds, batch):
        """Override to push frames to YouTube streaming."""
        log_string = super().write_results(idx, preds, batch)

        if hasattr(self, 'annotated_frame') and self.annotated_frame is not None:
            import cv2

            # Draw test indicator
            self._frame_count += 1
            text = f"TEST MODE | Frame: {self._frame_count} | Restarts: {self.stream_manager.restart_count}"
            cv2.putText(self.annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw counting line
            cv2.line(
                self.annotated_frame,
                tuple(COUNTING_LINE_START),
                tuple(COUNTING_LINE_END),
                (0, 255, 255), 2
            )

            # Push to YouTube streamer
            if self.stream_manager and self.stream_manager.is_running():
                try:
                    self.stream_manager.add_frame(self.annotated_frame)
                except Exception as e:
                    print(f"Error pushing frame to YouTube stream: {e}")

        return log_string


def main():
    parser = argparse.ArgumentParser(
        description='Test YouTube stream auto-restart (restarts every 5 minutes)'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Video file path or RTSP stream URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on: cuda or cpu (default: cuda)')
    parser.add_argument('--youtube-title', type=str,
                       default='[TEST] Auto-Restart Stream Test',
                       help='YouTube broadcast title')
    parser.add_argument('--youtube-privacy', type=str, default='unlisted',
                       choices=['public', 'private', 'unlisted'],
                       help='YouTube broadcast privacy (default: unlisted)')
    parser.add_argument('--client-secrets', type=str, default='client_secrets.json',
                       help='Path to YouTube API client secrets file')
    parser.add_argument('--restart-interval', type=int, default=300,
                       help='Restart interval in seconds (default: 300 = 5 minutes)')

    # Stream registration API arguments (optional)
    parser.add_argument('--registration-api-url', type=str, default=None,
                       help='URL for stream registration API (e.g., http://localhost:3000)')
    parser.add_argument('--registration-username', type=str, default=None,
                       help='Username for registration API Basic Auth')
    parser.add_argument('--registration-password', type=str, default=None,
                       help='Password for registration API Basic Auth')

    args = parser.parse_args()

    print("=" * 70)
    print("  YOUTUBE STREAM AUTO-RESTART TEST")
    print("=" * 70)
    print(f"  Restart interval: {args.restart_interval} seconds ({args.restart_interval // 60} minutes)")
    print(f"  Source: {args.source}")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  YouTube Title: {args.youtube_title}")
    print(f"  YouTube Privacy: {args.youtube_privacy}")
    if args.registration_api_url:
        print(f"  Registration API: {args.registration_api_url}")
    print("=" * 70)
    print()

    # Create configurations
    inference_config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=0.7,
        device=args.device,
        image_size=640
    )

    tracking_config = TrackingConfig()
    visualization_config = VisualizationConfig(
        counter_direction='north_enter',
        show_info_panel=True,
        show_counters=True
    )

    youtube_config = YouTubeStreamingConfig(
        enabled=True,
        client_secrets_file=args.client_secrets,
        broadcast_title=args.youtube_title,
        broadcast_description='Testing auto-restart functionality for 24/7 streaming',
        privacy_status=args.youtube_privacy,
        resolution='720p',
        video_bitrate=2500,
        auto_start=True
    )

    # Hydra config for BasePredictor
    hydra_config = {
        'model': args.model,
        'source': args.source,
        'conf': args.conf,
        'iou': 0.7,
        'device': args.device,
        'imgsz': [640, 640],
        'show': False,
        'save': False,
    }

    # Initialize registration client if credentials provided
    registration_client = None
    if args.registration_api_url and args.registration_username and args.registration_password:
        registration_client = StreamRegistrationClient(
            api_url=args.registration_api_url,
            username=args.registration_username,
            password=args.registration_password,
        )
        print("[TEST] Stream registration client initialized")

    # Initialize Stream Manager with test restart interval
    stream_manager = TestStreamRestartManager(
        youtube_config,
        restart_interval=args.restart_interval,
        registration_client=registration_client,
    )

    # Start YouTube stream
    print("[TEST] Starting initial YouTube stream...")
    if not stream_manager.start():
        print("[TEST] Failed to start YouTube stream. Exiting.")
        return 1

    # Wait for stream to initialize
    print("[TEST] Waiting for stream to initialize...")
    time.sleep(5)

    if not stream_manager.check_ffmpeg_health():
        print("[TEST] FFmpeg process is not healthy. Exiting.")
        return 1

    print("[TEST] Stream initialized and ready!")
    print()

    # Initialize inference engine
    engine = TestYouTubeStreamInference(
        stream_manager=stream_manager,
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )

    try:
        print("[TEST] Starting inference loop. Press Ctrl+C to stop.")
        print("[TEST] Watch the logs for restart events every 5 minutes.")
        print()

        engine()
        print("\n[TEST] Inference completed!")

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    except Exception as e:
        print(f"\n[TEST] Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream_manager.stop()

        # Print final statistics
        stats = stream_manager.get_stats()
        print()
        print("=" * 70)
        print("  TEST RESULTS")
        print("=" * 70)
        print(f"  Total frames streamed: {stats.get('frame_count', 0)}")
        print(f"  Total stream restarts: {stats.get('restart_count', 0)}")
        if stats.get('duration'):
            print(f"  Last stream duration: {stats['duration']:.1f} seconds")
            print(f"  Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"  Stream URL: {stats['stream_url']}")
        print("=" * 70)

        if stats.get('restart_count', 0) > 0:
            print("\n  SUCCESS: Auto-restart functionality is working!")
        else:
            print("\n  NOTE: No restarts occurred. Run for longer than 5 minutes to test.")


if __name__ == "__main__":
    main()
