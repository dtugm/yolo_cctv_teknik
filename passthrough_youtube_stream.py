#!/usr/bin/env python3
"""
Passthrough YouTube Live streaming script (no inference).

Reads frames from any video source (RTSP, file, webcam) and streams
directly to YouTube Live via FFmpeg/RTMP. No YOLO detection or tracking.

Features:
- Any OpenCV-compatible video source
- YouTube Live Streaming via FFmpeg/RTMP
- Auto-restart before YouTube's 12-hour limit (for 24/7 streaming)
- Stream registration for frontend listing
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time
import threading
import requests
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import YouTubeStreamingConfig
from src.streaming.youtube_streamer import YouTubeStreamer


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


class StreamRestartManager:
    """
    Manages automatic YouTube stream restarts to handle YouTube's 12-hour limit.
    Monitors stream duration and seamlessly restarts before hitting the limit.
    """

    # YouTube has a 12-hour limit; restart at 11h 55m to be safe
    MAX_STREAM_DURATION_SECONDS = 11 * 60 * 60 + 55 * 60  # 11 hours 55 minutes

    def __init__(self, youtube_config: YouTubeStreamingConfig, registration_client: StreamRegistrationClient = None):
        self.youtube_config = youtube_config
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
                    description=self.youtube_config.broadcast_description
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
                    description=self.youtube_config.broadcast_description
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


def main():
    parser = argparse.ArgumentParser(
        description='Passthrough YouTube Live Streaming (no inference)'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Video file path, RTSP stream URL, or device index (e.g. 0)')

    # YouTube streaming arguments
    parser.add_argument('--youtube-title', type=str,
                       default='CCTV Live Stream',
                       help='YouTube broadcast title')
    parser.add_argument('--youtube-description', type=str,
                       default='Live camera stream',
                       help='YouTube broadcast description')
    parser.add_argument('--youtube-privacy', type=str, default='unlisted',
                       choices=['public', 'private', 'unlisted'],
                       help='YouTube broadcast privacy (default: unlisted)')
    parser.add_argument('--youtube-resolution', type=str, default='720p',
                       choices=['480p', '720p', '1080p'],
                       help='YouTube stream resolution (default: 720p)')
    parser.add_argument('--youtube-bitrate', type=int, default=2500,
                       help='YouTube stream video bitrate in kbps (default: 2500)')
    parser.add_argument('--client-secrets', type=str, default='client_secrets.json',
                       help='Path to YouTube API client secrets file')
    parser.add_argument('--manual-transition', action='store_true',
                       help='Manually transition to live (instead of automatic)')

    # Stream registration arguments
    parser.add_argument('--registration-api-url', type=str, default=None,
                       help='URL of stream registration service. If unset, registration disabled.')
    parser.add_argument('--registration-username', type=str, default=None,
                       help='Basic auth username for registration API')
    parser.add_argument('--registration-password', type=str, default=None,
                       help='Basic auth password for registration API')

    # Streaming control
    parser.add_argument('--target-fps', type=float, default=30.0,
                       help='Target frames per second for streaming (default: 30)')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

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

    # Create YouTube config
    youtube_config = YouTubeStreamingConfig(
        enabled=True,
        client_secrets_file=args.client_secrets,
        broadcast_title=args.youtube_title,
        broadcast_description=args.youtube_description,
        privacy_status=args.youtube_privacy,
        resolution=args.youtube_resolution,
        video_bitrate=args.youtube_bitrate,
        auto_start=not args.manual_transition,
    )

    print("==============================================================")
    print("  Passthrough YouTube Live Stream (no inference)")
    print("==============================================================")
    print(f"   Source: {args.source}")
    print(f"   YouTube Title: {args.youtube_title}")
    print(f"   YouTube Privacy: {args.youtube_privacy}")
    print(f"   YouTube Resolution: {args.youtube_resolution}")
    print(f"   Target FPS: {args.target_fps}")
    if args.manual_transition:
        print(f"   Transition Mode: Manual")
    print()

    # Open video source
    # Try parsing as integer (webcam device index), otherwise use as string (file/RTSP)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return 1

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Source info: {source_width}x{source_height} @ {source_fps:.1f} FPS")
    print()

    # Initialize Stream Manager
    stream_manager = StreamRestartManager(youtube_config, registration_client=registration_client)
    print("   Auto-restart: Enabled (restarts before 12-hour YouTube limit)")
    if registration_client:
        print("   Stream Registration: Enabled")

    # Start YouTube stream
    if not stream_manager.start():
        print("Failed to start YouTube stream. Exiting.")
        cap.release()
        return 1

    # Wait for stream to initialize
    print("Waiting for YouTube stream to initialize...")
    time.sleep(5)

    if not stream_manager.check_ffmpeg_health():
        print("FFmpeg process is not healthy. Exiting.")
        stream_manager.stop()
        cap.release()
        return 1

    print("YouTube stream initialized and ready for frames")

    frame_interval = 1.0 / args.target_fps
    frame_count = 0

    try:
        print("Starting passthrough streaming to YouTube...")
        print("   Press Ctrl+C to stop streaming")
        print()

        while cap.isOpened():
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                # For files, this means end of video
                if isinstance(source, str) and not source.startswith("rtsp"):
                    print("End of video file reached.")
                    break
                # For RTSP, try to reconnect
                print("Failed to read frame, attempting reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    print("Reconnect failed. Exiting.")
                    break
                continue

            stream_manager.add_frame(frame)
            frame_count += 1

            if args.verbose and frame_count % 300 == 0:
                print(f"[Passthrough] Frames streamed: {frame_count}")

            # Check stream health periodically
            if frame_count % 900 == 0:
                if not stream_manager.is_running():
                    print("Stream is no longer running. Exiting.")
                    break

            # FPS throttling
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during streaming: {e}")
    finally:
        cap.release()
        stream_manager.stop()

        # Print final statistics
        stats = stream_manager.get_stats()
        print(f"\nFinal Stream Statistics:")
        print(f"   Total frames streamed: {frame_count}")
        print(f"   Total stream restarts: {stats.get('restart_count', 0)}")
        if stats.get('duration'):
            print(f"   Last stream duration: {stats['duration']:.1f} seconds")
            print(f"   Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"   Stream URL: {stats['stream_url']}")


if __name__ == "__main__":
    main()
