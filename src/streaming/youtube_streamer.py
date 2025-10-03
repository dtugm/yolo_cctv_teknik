"""YouTube Live streaming for YOLO inference results."""

import os
import threading
import time
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

from src.config.settings import YouTubeStreamingConfig


class YouTubeStreamer:
    """
    YouTube Live streaming server for broadcasting annotated video frames.
    Handles authentication, stream creation, FFmpeg encoding, and cleanup.
    """
    
    # YouTube API scopes
    SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
    
    def __init__(self, config: Optional[YouTubeStreamingConfig] = None):
        """
        Initialize YouTube streamer.
        
        Args:
            config: YouTube streaming configuration
            
        Raises:
            ImportError: If required YouTube API libraries are not installed
            FileNotFoundError: If client secrets file is not found
        """
        if not YOUTUBE_AVAILABLE:
            raise ImportError(
                "YouTube API libraries not installed. Run: "
                "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )
        
        self.config = config or YouTubeStreamingConfig()
        
        # Validate client secrets file
        if not os.path.exists(self.config.client_secrets_file):
            raise FileNotFoundError(
                f"Client secrets file not found: {self.config.client_secrets_file}\n"
                "Please download from Google Cloud Console and place in project root."
            )
        
        # YouTube API client
        self.youtube = None
        self.credentials = None
        
        # Stream state
        self.running = False
        self.broadcast_id: Optional[str] = None
        self.stream_id: Optional[str] = None
        self.rtmp_url: Optional[str] = None
        self.stream_name: Optional[str] = None
        
        # FFmpeg process
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.ffmpeg_stdin = None
        self.ffmpeg_stderr_thread = None
        
        # Thread-safe frame storage
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        
        # Timing
        self.start_time = None
        self.last_frame_time = 0.0
        self.target_frame_interval = 1.0 / 30.0  # 30 FPS default
        
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API using OAuth 2.0.
        
        Returns:
            True if authentication successful, False otherwise
        """
        creds = None
        token_file = 'youtube_token.pickle'
        
        # Load existing credentials
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Failed to refresh credentials: {e}")
                    creds = None
            
            if not creds:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config.client_secrets_file, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"Authentication failed: {e}")
                    return False
            
            # Save credentials for next run
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build YouTube API client
        try:
            self.youtube = build('youtube', 'v3', credentials=creds)
            self.credentials = creds
            print("✅ YouTube API authentication successful")
            return True
        except Exception as e:
            print(f"Failed to build YouTube API client: {e}")
            return False
    
    def create_broadcast(self) -> bool:
        """
        Create a YouTube Live broadcast.
        
        Returns:
            True if broadcast created successfully, False otherwise
        """
        if not self.youtube:
            print("YouTube API not authenticated")
            return False
        
        try:
            # Schedule start time (immediate)
            scheduled_start_time = datetime.now(timezone.utc).isoformat()
            
            request = self.youtube.liveBroadcasts().insert(
                part="snippet,status,contentDetails",
                body={
                    "snippet": {
                        "title": self.config.broadcast_title,
                        "description": self.config.broadcast_description,
                        "scheduledStartTime": scheduled_start_time
                    },
                    "status": {
                        "privacyStatus": self.config.privacy_status,
                        "selfDeclaredMadeForKids": False
                    },
                    "contentDetails": {
                        "enableAutoStart": self.config.auto_start,
                        "enableAutoStop": False,  # We'll handle this manually
                        "enableDvr": True,
                        "enableContentEncryption": False,
                        "startWithSlate": False,
                        "recordFromStart": True,
                        "enableMonitorStream": True  # Required for testing phase
                    }
                }
            )
            
            response = request.execute()
            self.broadcast_id = response['id']
            
            print(f"✅ Created YouTube broadcast: {self.broadcast_id}")
            print(f"   Title: {self.config.broadcast_title}")
            print(f"   Privacy: {self.config.privacy_status}")
            
            return True
            
        except Exception as e:
            print(f"Failed to create broadcast: {e}")
            return False
    
    def create_stream(self) -> bool:
        """
        Create a YouTube Live stream.
        
        Returns:
            True if stream created successfully, False otherwise
        """
        if not self.youtube:
            print("YouTube API not authenticated")
            return False
        
        try:
            request = self.youtube.liveStreams().insert(
                part="snippet,cdn",
                body={
                    "snippet": {
                        "title": f"Stream for {self.config.broadcast_title}"
                    },
                    "cdn": {
                        "ingestionType": "rtmp",
                        "resolution": self.config.resolution,
                        "frameRate": self.config.frame_rate
                    }
                }
            )
            
            response = request.execute()
            self.stream_id = response['id']
            
            # Extract RTMP details
            ingestion_info = response['cdn']['ingestionInfo']
            ingestion_address = ingestion_info['ingestionAddress']
            stream_name = ingestion_info['streamName']
            
            self.rtmp_url = f"{ingestion_address}/{stream_name}"
            self.stream_name = stream_name
            
            print(f"✅ Created YouTube stream: {self.stream_id}")
            print(f"   RTMP URL: {ingestion_address}")
            print(f"   Stream Name: {stream_name}")
            print(f"   Full RTMP URL: {self.rtmp_url}")
            print(f"   Resolution: {self.config.resolution}")
            print(f"   Frame rate: {self.config.frame_rate}")
            
            # Validate RTMP URL format
            self._validate_rtmp_url()
            
            # Test stream key validity
            self._test_stream_key_validity()
            
            return True
            
        except Exception as e:
            print(f"Failed to create stream: {e}")
            return False
    
    def bind_broadcast_to_stream(self) -> bool:
        """
        Bind the broadcast to the stream.
        
        Returns:
            True if binding successful, False otherwise
        """
        if not self.youtube or not self.broadcast_id or not self.stream_id:
            print("Missing broadcast or stream for binding")
            return False
        
        try:
            request = self.youtube.liveBroadcasts().bind(
                part="id,contentDetails",
                id=self.broadcast_id,
                streamId=self.stream_id
            )
            
            response = request.execute()
            print(f"✅ Bound broadcast to stream successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to bind broadcast to stream: {e}")
            return False
    
    def _get_resolution_dimensions(self) -> tuple:
        """Get width and height from resolution string."""
        resolution_map = {
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080)
        }
        return resolution_map.get(self.config.resolution, (1280, 720))
    
    def _validate_rtmp_url(self) -> None:
        """Validate RTMP URL format for YouTube Live."""
        if not self.rtmp_url:
            print("❌ RTMP URL is empty")
            return
            
        print("🔍 Validating RTMP URL format...")
        
        # Check if URL starts with rtmp://
        if not self.rtmp_url.startswith('rtmp://'):
            print("❌ RTMP URL must start with rtmp://")
            return
            
        # Check if URL contains required components (YouTube uses /live2 now)
        if ('/live/' not in self.rtmp_url) and ('/live2/' not in self.rtmp_url):
            print("❌ RTMP URL missing /live/ or /live2/ component")
            return
            
        # Check if stream name is present
        if not self.stream_name:
            print("❌ Stream name is missing")
            return
            
        # Check URL length (YouTube has limits)
        if len(self.rtmp_url) > 500:
            print("⚠️  RTMP URL is very long, might cause issues")
            
        print("✅ RTMP URL format appears valid")
        print(f"   URL length: {len(self.rtmp_url)} characters")
        print(f"   Stream name length: {len(self.stream_name)} characters")
        
        # Check if stream key looks valid (YouTube stream keys are typically 4-4-4-4-4 format)
        if len(self.stream_name) != 24 or self.stream_name.count('-') != 4:
            print(f"⚠️  Stream key format unusual: {self.stream_name}")
        else:
            print(f"✅ Stream key format looks correct: {self.stream_name}")
    
    def _test_stream_key_validity(self) -> bool:
        """Test if the stream key is valid by checking stream status."""
        try:
            print("🔑 Testing stream key validity...")
            
            # Check if stream exists and is accessible
            request = self.youtube.liveStreams().list(
                part="status,cdn",
                id=self.stream_id
            )
            response = request.execute()
            
            if response.get('items'):
                stream_info = response['items'][0]
                stream_status = stream_info['status']['streamStatus']
                print(f"📊 Stream status: {stream_status}")
                
                # Check if stream is in a valid state
                if stream_status in ['created', 'ready', 'active']:
                    print("✅ Stream key appears valid")
                    return True
                else:
                    print(f"⚠️  Stream in unexpected state: {stream_status}")
                    return False
            else:
                print("❌ Stream not found - stream key may be invalid")
                return False
                
        except Exception as e:
            print(f"❌ Error testing stream key: {e}")
            return False
    
    def _start_ffmpeg(self) -> bool:
        """
        Start FFmpeg process for encoding and streaming.
        
        Returns:
            True if FFmpeg started successfully, False otherwise
        """
        if not self.rtmp_url:
            print("No RTMP URL available for FFmpeg")
            return False
        
        try:
            width, height = self._get_resolution_dimensions()

            # Build two candidate commands: prefer macOS hardware encoder if available, then fall back to libx264
            # Common input args (raw BGR frames via stdin) + silent audio
            common_input_args = [
                '-y',
                '-fflags', 'nobuffer',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', '30',
                '-use_wallclock_as_timestamps', '1',
                '-i', '-',
                # Add a silent audio source so YouTube sees both audio and video
                '-f', 'lavfi',
                '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            ]

            # Common output args
            common_output_args = [
                '-pix_fmt', 'yuv420p',
                '-g', '60',
                '-keyint_min', '30',
                '-sc_threshold', '0',
                '-tune', 'zerolatency',
                # Audio encoding
                '-c:a', 'aac',
                '-b:a', f'{self.config.audio_bitrate}k',
                '-ar', '44100',
                '-ac', '2',
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                '-rtmp_live', '1',
                '-rtmp_buffer', '1000',
                '-connect_timeout', '10',
                '-rw_timeout', '5000000',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '2',
                '-flush_packets', '1',
            ]

            # Hardware encoder (macOS VideoToolbox)
            cmd_videotoolbox = (
                ['ffmpeg']
                + common_input_args
                + [
                    '-c:v', 'h264_videotoolbox',
                    '-b:v', f'{self.config.video_bitrate}k',
                    '-maxrate', f'{self.config.video_bitrate}k',
                    '-bufsize', f'{self.config.video_bitrate * 2}k',
                    '-profile:v', 'main',
                ]
                + [
                    # Map video from input 0, audio from input 1
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                ]
                + common_output_args
                + [self.rtmp_url]
            )

            # Software encoder (libx264)
            cmd_libx264 = (
                ['ffmpeg']
                + common_input_args
                + [
                    '-c:v', 'libx264',
                    '-preset', self.config.ffmpeg_preset,
                    '-crf', str(self.config.ffmpeg_crf),
                    '-maxrate', f'{self.config.video_bitrate}k',
                    '-bufsize', f'{self.config.video_bitrate * 2}k',
                    '-profile:v', 'main',
                    '-level', '4.1',
                ]
                + [
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                ]
                + common_output_args
                + [self.rtmp_url]
            )

            # Try hardware first, then fallback to software
            candidate_cmds = [cmd_videotoolbox, cmd_libx264]

            last_error = None
            for cmd in candidate_cmds:
                print(f"🔧 Starting FFmpeg with command: {' '.join(cmd)}")
                
                self.ffmpeg_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                self.ffmpeg_stdin = self.ffmpeg_process.stdin

                print("⏳ Waiting for FFmpeg to initialize and connect to RTMP...")
                time.sleep(3)

                if self.ffmpeg_process.poll() is None:
                    # Looks good; continue with this process
                    break
                else:
                    try:
                        last_error = self.ffmpeg_process.stderr.read().decode()
                        print(f"❌ FFmpeg failed to start with this encoder: {last_error}")
                    except Exception:
                        print("❌ FFmpeg failed to start (unable to read stderr)")
                    # Reset and try next candidate
                    self.ffmpeg_process = None
                    self.ffmpeg_stdin = None
            
            # If still no running process, fail
            if not self.ffmpeg_process or self.ffmpeg_process.poll() is not None:
                print("❌ FFmpeg could not be started with available encoders")
                if last_error:
                    print(f"   Last error: {last_error}")
                return False
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.ffmpeg_stdin = self.ffmpeg_process.stdin
            
            # Give FFmpeg more time to start and connect to RTMP
            print("⏳ Waiting for FFmpeg to initialize and connect to RTMP...")
            time.sleep(3)
            
            # Check if FFmpeg started successfully
            if self.ffmpeg_process.poll() is not None:
                # FFmpeg process terminated immediately
                try:
                    stderr_output = self.ffmpeg_process.stderr.read().decode()
                    print(f"❌ FFmpeg failed to start: {stderr_output}")
                except:
                    print("❌ FFmpeg failed to start (unable to read stderr)")
                return False
            
            # Additional check: verify FFmpeg is still running after initial wait
            time.sleep(2)
            if self.ffmpeg_process.poll() is not None:
                print("❌ FFmpeg process died during initialization")
                try:
                    stderr_output = self.ffmpeg_process.stderr.read().decode()
                    print(f"❌ FFmpeg stderr: {stderr_output}")
                except:
                    pass
                return False
            
            # Start monitoring FFmpeg stderr for errors
            self._start_ffmpeg_monitoring()
            
            print(f"✅ FFmpeg started for streaming")
            print(f"   Resolution: {width}x{height}")
            print(f"   Bitrate: {self.config.video_bitrate}k")
            print(f"   Preset: {self.config.ffmpeg_preset}")
            print(f"   RTMP URL: {self.rtmp_url}")
            print(f"   Full FFmpeg command: {' '.join(cmd)}")
            
            # Test RTMP connection with a simple probe
            self._test_rtmp_connection()
            
            # Test RTMP connection with minimal FFmpeg command
            print("🔍 Testing RTMP connection with minimal parameters...")
            minimal_test_success = self._test_rtmp_with_minimal_ffmpeg()
            if not minimal_test_success:
                print("❌ Minimal RTMP test failed - stream key or URL may be invalid")
                return False
            
            # Test RTMP connection with simple FFmpeg command
            rtmp_test_success = self._test_rtmp_connection_simple()
            if not rtmp_test_success:
                print("⚠️  Simple RTMP connection test failed, but continuing...")
            
            # Send a test frame to verify connection
            self._send_test_frame()
            
            return True
            
        except Exception as e:
            print(f"Failed to start FFmpeg: {e}")
            return False
    
    def check_stream_status(self) -> Optional[str]:
        """
        Check the status of the bound stream.
        
        Returns:
            Stream status if available, None otherwise
        """
        if not self.youtube or not self.stream_id:
            return None
        
        try:
            request = self.youtube.liveStreams().list(
                part="status",
                id=self.stream_id
            )
            
            response = request.execute()
            
            if response.get('items'):
                stream_status = response['items'][0]['status']['streamStatus']
                print(f"📊 Stream status: {stream_status}")
                return stream_status
            else:
                print("⚠️  Stream not found")
                return None
                
        except Exception as e:
            print(f"Failed to check stream status: {e}")
            return None
    
    def _test_rtmp_connection(self) -> None:
        """Test RTMP connection with a simple probe."""
        try:
            print("🔍 Testing RTMP connection...")
            # Use FFprobe to test the RTMP connection
            probe_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-timeout', '10000000',  # 10 second timeout
                self.rtmp_url
            ]
            
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                print("✅ RTMP connection test successful")
            else:
                print(f"⚠️  RTMP connection test failed: {result.stderr}")
                print("   This might be normal - YouTube may not accept probes")
                
        except subprocess.TimeoutExpired:
            print("⚠️  RTMP connection test timed out")
        except Exception as e:
            print(f"⚠️  RTMP connection test error: {e}")
    
    def _test_rtmp_with_minimal_ffmpeg(self) -> bool:
        """Test RTMP connection with absolute minimal FFmpeg command."""
        try:
            print("🧪 Testing RTMP with minimal FFmpeg command...")
            
            # Create a 1-second test video with minimal parameters
            test_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'lavfi',
                '-i', 'color=c=black:size=640x480:duration=1',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '28',
                '-pix_fmt', 'yuv420p',
                '-f', 'flv',
                '-t', '1',
                self.rtmp_url
            ]
            
            print(f"🔍 Minimal test command: {' '.join(test_cmd)}")
            
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"📊 Return code: {result.returncode}")
            if result.stderr:
                print(f"📊 Stderr: {result.stderr}")
            if result.stdout:
                print(f"📊 Stdout: {result.stdout}")
            
            if result.returncode == 0:
                print("✅ Minimal RTMP test successful!")
                return True
            else:
                print("❌ Minimal RTMP test failed")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  Minimal RTMP test timed out")
            return False
        except Exception as e:
            print(f"⚠️  Minimal RTMP test error: {e}")
            return False
    
    def _test_rtmp_connection_simple(self) -> bool:
        """Test RTMP connection with a simple FFmpeg command."""
        try:
            print("🧪 Testing RTMP connection with simple FFmpeg command...")
            
            # Create a simple test video (1 second of black frames)
            test_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'lavfi',
                '-i', 'testsrc=duration=1:size=640x480:rate=30',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-t', '1',  # 1 second duration
                '-f', 'flv',
                self.rtmp_url
            ]
            
            print(f"🔍 Test command: {' '.join(test_cmd)}")
            
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ RTMP connection test successful!")
                return True
            else:
                print(f"❌ RTMP connection test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  RTMP connection test timed out")
            return False
        except Exception as e:
            print(f"⚠️  RTMP connection test error: {e}")
            return False
    
    def _send_test_frame(self) -> None:
        """Send a test frame to verify RTMP connection."""
        try:
            print("🧪 Sending test frame to verify RTMP connection...")
            width, height = self._get_resolution_dimensions()
            
            # Create a simple test frame (solid color)
            test_frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)  # Green frame
            
            # Add some text to the frame
            cv2.putText(test_frame, 'TEST FRAME', (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(test_frame, f'{width}x{height}', (50, height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Send multiple test frames
            for i in range(10):
                if self.ffmpeg_stdin and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    frame_bytes = test_frame.tobytes()
                    self.ffmpeg_stdin.write(frame_bytes)
                    self.ffmpeg_stdin.flush()
                    time.sleep(0.033)  # ~30 FPS
                else:
                    print("⚠️  FFmpeg not ready for test frame")
                    break
            
            print("✅ Test frames sent")
            
            # Check stream status after sending test frames
            time.sleep(2)
            print("🔍 Checking stream status after test frames...")
            stream_status = self.check_stream_status()
            
            if stream_status == 'active':
                print("🎉 Stream became active after test frames!")
            else:
                print(f"📊 Stream status after test frames: {stream_status}")
            
        except Exception as e:
            print(f"⚠️  Error sending test frame: {e}")
    
    def _start_ffmpeg_monitoring(self) -> None:
        """Start monitoring FFmpeg stderr for errors."""
        def monitor_stderr():
            try:
                while self.running and self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    line = self.ffmpeg_process.stderr.readline()
                    if line:
                        error_msg = line.decode().strip()
                        if error_msg:
                            # Always show important messages
                            if any(keyword in error_msg.lower() for keyword in [
                                'error', 'failed', 'connection', 'rtmp', 'timeout', 'refused'
                            ]):
                                print(f"🚨 FFmpeg Error: {error_msg}")
                            elif not error_msg.startswith('frame=') and not error_msg.startswith('size='):
                                print(f"🔍 FFmpeg: {error_msg}")
            except Exception as e:
                print(f"Error monitoring FFmpeg stderr: {e}")
        
        self.ffmpeg_stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        self.ffmpeg_stderr_thread.start()
    
    def check_ffmpeg_health(self) -> bool:
        """Check if FFmpeg process is healthy and connected."""
        if not self.ffmpeg_process:
            print("⚠️  FFmpeg process not started")
            return False
            
        if self.ffmpeg_process.poll() is not None:
            print(f"⚠️  FFmpeg process terminated with code {self.ffmpeg_process.returncode}")
            # Try to read any remaining stderr
            try:
                stderr_output = self.ffmpeg_process.stderr.read().decode()
                if stderr_output:
                    print(f"🚨 FFmpeg stderr: {stderr_output}")
            except:
                pass
            return False
            
        return True
    
    def start(self) -> bool:
        """
        Start the YouTube streaming process.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        if not self.config.enabled:
            print("YouTube streaming is disabled in config")
            return False
        
        print("🔴 Starting YouTube Live Stream...")
        
        # Step 1: Authenticate
        if not self.authenticate():
            return False
        
        # Step 2: Create broadcast
        if not self.create_broadcast():
            return False
        
        # Step 3: Create stream
        if not self.create_stream():
            return False
        
        # Step 4: Bind broadcast to stream
        if not self.bind_broadcast_to_stream():
            return False
        
        # Step 5: Start FFmpeg
        if not self._start_ffmpeg():
            return False
        
        # Step 6: Do not block waiting for active; return so frames can be sent
        print("ℹ️ Not waiting for 'active' status; will go live when frames arrive.")
        # Health check once before returning
        if not self.check_ffmpeg_health():
            print("❌ FFmpeg process died during initialization")
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # Get the stream URL for user
        if self.broadcast_id:
            stream_url = f"https://www.youtube.com/watch?v={self.broadcast_id}"
            print(f"🌍 YouTube Live Stream URL: {stream_url}")
        
        print("✅ YouTube streaming setup completed!")
        print("   Note: Stream will transition to LIVE automatically when data is received")
        
        # Print debug info
        self.debug_stream_info()
        
        return True
    
    def transition_broadcast_to_live(self) -> bool:
        """
        Transition broadcast status following YouTube's lifecycle: ready -> testing -> live.
        
        Returns:
            True if transition successful, False otherwise
        """
        if not self.youtube or not self.broadcast_id:
            return False
        
        try:
            # First, check the current status
            status_request = self.youtube.liveBroadcasts().list(
                part='status',
                id=self.broadcast_id
            )
            status_response = status_request.execute()
            
            if not status_response.get('items'):
                print("⚠️  Broadcast not found")
                return False
            
            current_status = status_response['items'][0]['status']['lifeCycleStatus']
            print(f"📊 Current broadcast status: {current_status}")
            
            # Check stream status first (required by YouTube API docs)
            stream_status = self.check_stream_status()
            
            # Only transition if not already live
            if current_status == 'live':
                print("✅ Broadcast is already LIVE")
                return True
            elif current_status == 'ready':
                # First transition to testing (required for monitor stream)
                if stream_status != 'active':
                    print(f"⚠️  Stream status is {stream_status}, not active. Cannot transition yet.")
                    return False
                    
                print("🔄 Transitioning to testing phase...")
                request = self.youtube.liveBroadcasts().transition(
                    broadcastStatus='testing',
                    id=self.broadcast_id,
                    part='id,status'
                )
                
                response = request.execute()
                
                print("response - transtion to test", response)
                print("✅ Broadcast transitioned to TESTING")
                
                # Wait a moment for testing to stabilize
                time.sleep(3)
                
                # Now transition to live
                print("🔄 Transitioning to live phase...")
                request = self.youtube.liveBroadcasts().transition(
                    broadcastStatus='live',
                    id=self.broadcast_id,
                    part='id,status'
                )
                
                response = request.execute()
                print("response - transtion to live", response)
                
                print("✅ Broadcast transitioned to LIVE")
                return True
                
            elif current_status == 'testing':
                # Already in testing, transition to live
                if stream_status != 'active':
                    print(f"⚠️  Stream status is {stream_status}, not active. Cannot transition yet.")
                    return False
                    
                print("🔄 Transitioning from testing to live...")
                request = self.youtube.liveBroadcasts().transition(
                    broadcastStatus='live',
                    id=self.broadcast_id,
                    part='id,status'
                )
                
                response = request.execute()
                print("✅ Broadcast transitioned to LIVE")
                return True
                
            elif current_status in ['testStarting', 'liveStarting']:
                print(f"📊 Broadcast status: {current_status} - transition in progress")
                print("   Waiting for transition to complete...")
                return True  # Don't fail, transition is in progress
            else:
                print(f"⚠️  Cannot transition from status: {current_status}")
                print("   Stream will go live automatically when data is received")
                return True  # Don't fail, YouTube will handle it
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid transition" in error_msg:
                print("⚠️  Invalid transition - stream may already be live or transitioning")
                print("   This is normal, YouTube will handle the transition automatically")
                return True  # Don't fail, this is often expected
            else:
                print(f"Failed to transition broadcast to live: {e}")
                return False
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the YouTube stream.
        
        Args:
            frame: OpenCV frame (BGR format)
        """
        # Debug: Log every 30th frame attempt
        if not hasattr(self, '_add_frame_debug_count'):
            self._add_frame_debug_count = 0
        self._add_frame_debug_count += 1
        
        if self._add_frame_debug_count % 30 == 0:
            print(f"🎬 add_frame called {self._add_frame_debug_count} times")
            print(f"   Frame shape: {frame.shape}")
            print(f"   Frame dtype: {frame.dtype}")
            print(f"   Running: {self.running}")
            print(f"   FFmpeg stdin: {self.ffmpeg_stdin is not None}")
            print(f"   FFmpeg process: {self.ffmpeg_process is not None}")
            if self.ffmpeg_process:
                print(f"   FFmpeg alive: {self.ffmpeg_process.poll() is None}")
        
        if not self.running:
            if self._add_frame_debug_count % 30 == 0:
                print("⚠️  YouTube streamer not running")
            return
            
        if not self.ffmpeg_stdin:
            if self._add_frame_debug_count % 30 == 0:
                print("⚠️  FFmpeg stdin not available")
            return
            
        # Check FFmpeg health
        if not self.check_ffmpeg_health():
            self.running = False
            return
        
        try:
            current_time = time.time()
            
            # Frame rate limiting
            if current_time - self.last_frame_time < self.target_frame_interval:
                if self._add_frame_debug_count % 30 == 0:
                    print(f"⏱️  Frame rate limited: {current_time - self.last_frame_time:.3f}s < {self.target_frame_interval:.3f}s")
                return
            
            with self.lock:
                # Ensure frame is in correct format (BGR, uint8)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Ensure frame has 3 channels (BGR)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Resize frame to target resolution
                width, height = self._get_resolution_dimensions()
                resized_frame = cv2.resize(frame, (width, height))
                
                # Ensure frame is contiguous in memory
                if not resized_frame.flags['C_CONTIGUOUS']:
                    resized_frame = np.ascontiguousarray(resized_frame)
                
                # Write frame to FFmpeg stdin
                try:
                    frame_bytes = resized_frame.tobytes()
                    self.ffmpeg_stdin.write(frame_bytes)
                    self.ffmpeg_stdin.flush()
                    
                    self.frame_count += 1
                    self.last_frame_time = current_time
                    
                    # Debug: Print frame info every 30 frames
                    if self.frame_count % 30 == 0:
                        print(f"📹 YouTube streamer received frame {self.frame_count} ({len(frame_bytes)} bytes)")
                        print(f"   Frame shape: {resized_frame.shape}, FFmpeg process alive: {self.ffmpeg_process.poll() is None}")
                        print(f"   FFmpeg stdin available: {self.ffmpeg_stdin is not None}")
                        print(f"   Running state: {self.running}")
                        print(f"   Frame contiguous: {resized_frame.flags['C_CONTIGUOUS']}")
                    
                    # Transition to live after a few frames (if auto_start enabled)
                    if (self.frame_count == 10 and self.config.auto_start and 
                        self.broadcast_id and not hasattr(self, '_transitioned_to_live')):
                        print("🔄 Attempting to transition to live after 10 frames...")
                        # Give more time for stream to stabilize
                        time.sleep(5)
                        if self.transition_broadcast_to_live():
                            self._transitioned_to_live = True
                        else:
                            print("⚠️  Auto transition failed, stream will go live when YouTube detects data")
                    
                    # Check max duration
                    if (self.start_time and 
                        current_time - self.start_time > self.config.max_stream_duration):
                        print("⚠️  Maximum stream duration reached, stopping...")
                        self.stop()
                        
                except BrokenPipeError:
                    print("⚠️  FFmpeg pipe broken, stream may have ended")
                    self.running = False
                except Exception as e:
                    print(f"⚠️  Error writing frame to FFmpeg: {e}")
                    # Try to restart FFmpeg if it's not responding
                    if not self.check_ffmpeg_health():
                        print("🔄 FFmpeg process died, attempting restart...")
                        self.running = False
                
        except Exception as e:
            print(f"Error adding frame to YouTube stream: {e}")
    
    def end_broadcast(self) -> bool:
        """
        End the YouTube broadcast.
        
        Returns:
            True if broadcast ended successfully, False otherwise
        """
        if not self.youtube or not self.broadcast_id:
            return False
        
        try:
            # Check current lifecycle status to avoid invalid transitions
            status_request = self.youtube.liveBroadcasts().list(
                part='status',
                id=self.broadcast_id
            )
            status_response = status_request.execute()
            if not status_response.get('items'):
                print("⚠️  Broadcast not found when ending")
                return False

            life_cycle = status_response['items'][0]['status']['lifeCycleStatus']
            print(f"📊 Broadcast lifecycle before end: {life_cycle}")

            if life_cycle in ['complete', 'revoked', 'testStarting', 'liveStarting']:
                print("ℹ️ Broadcast already ending/ended or in transitional state; skipping end transition")
                return True

            # Only attempt to complete if in testing/live/ready
            if life_cycle in ['testing', 'live', 'ready']:
                request = self.youtube.liveBroadcasts().transition(
                    broadcastStatus='complete',
                    id=self.broadcast_id,
                    part='id,status'
                )
                response = request.execute()
                print("✅ YouTube broadcast ended")
                return True
            else:
                print(f"ℹ️ Skipping end transition from state: {life_cycle}")
                return True

        except Exception as e:
            print(f"Failed to end broadcast: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the YouTube streaming process."""
        if not self.running:
            return
        
        print("🛑 Stopping YouTube streaming...")
        
        self.running = False
        
        # Stop FFmpeg
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_stdin:
                    self.ffmpeg_stdin.close()
                
                # Wait for FFmpeg to finish
                self.ffmpeg_process.wait(timeout=10)
                print("✅ FFmpeg stopped")
                
            except subprocess.TimeoutExpired:
                print("⚠️  FFmpeg timeout, force killing...")
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
            except Exception as e:
                print(f"Error stopping FFmpeg: {e}")
            
            self.ffmpeg_process = None
            self.ffmpeg_stdin = None
        
        # End broadcast
        if self.config.auto_end and self.broadcast_id:
            self.end_broadcast()
        
        # Print stats
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"📊 Stream stats:")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Frames sent: {self.frame_count}")
            print(f"   Average FPS: {self.frame_count / duration:.1f}")
        
        print("✅ YouTube streaming stopped")
    
    def get_stream_url(self) -> Optional[str]:
        """
        Get the YouTube stream URL for viewers.
        
        Returns:
            YouTube watch URL if available, None otherwise
        """
        if self.broadcast_id:
            return f"https://www.youtube.com/watch?v={self.broadcast_id}"
        return None
    
    def is_running(self) -> bool:
        """Check if the stream is currently running."""
        return self.running and self.ffmpeg_process is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get streaming statistics.
        
        Returns:
            Dictionary with stream statistics
        """
        stats = {
            'running': self.running,
            'frame_count': self.frame_count,
            'broadcast_id': self.broadcast_id,
            'stream_id': self.stream_id,
            'stream_url': self.get_stream_url()
        }
        
        if self.start_time:
            duration = time.time() - self.start_time
            stats.update({
                'duration': duration,
                'fps': self.frame_count / duration if duration > 0 else 0
            })
        
        return stats
    
    def force_transition_to_live(self) -> bool:
        """
        Force transition to live status (for manual control).
        
        Returns:
            True if transition successful, False otherwise
        """
        print("🔄 Attempting to force transition to live...")
        return self.transition_broadcast_to_live()
    
    def debug_stream_info(self) -> None:
        """Print detailed stream debugging information."""
        print("\n🔍 === STREAM DEBUG INFO ===")
        
        # Check broadcast status
        if self.broadcast_id:
            try:
                status_request = self.youtube.liveBroadcasts().list(
                    part='status',
                    id=self.broadcast_id
                )
                status_response = status_request.execute()
                
                if status_response.get('items'):
                    broadcast_status = status_response['items'][0]['status']['lifeCycleStatus']
                    print(f"📊 Broadcast Status: {broadcast_status}")
                else:
                    print("⚠️  Broadcast not found")
            except Exception as e:
                print(f"❌ Error checking broadcast status: {e}")
        
        # Check stream status
        stream_status = self.check_stream_status()
        print(f"📊 Stream Status: {stream_status}")
        
        # Check FFmpeg health
        ffmpeg_healthy = self.check_ffmpeg_health()
        print(f"📊 FFmpeg Health: {'✅ Healthy' if ffmpeg_healthy else '❌ Unhealthy'}")
        
        # Check frame count
        print(f"📊 Frames Sent: {self.frame_count}")
        
        # Check RTMP URL
        print(f"📊 RTMP URL: {self.rtmp_url}")
        
        # Check broadcast-stream binding
        self.check_broadcast_stream_binding()
        
        print("🔍 === END DEBUG INFO ===\n")
    
    def check_broadcast_stream_binding(self) -> bool:
        """Check if broadcast and stream are properly bound."""
        try:
            print("🔗 Checking broadcast-stream binding...")
            
            if not self.broadcast_id:
                print("❌ No broadcast ID")
                return False
                
            # Get broadcast details
            request = self.youtube.liveBroadcasts().list(
                part='contentDetails',
                id=self.broadcast_id
            )
            response = request.execute()
            
            if not response.get('items'):
                print("❌ Broadcast not found")
                return False
                
            broadcast = response['items'][0]
            content_details = broadcast.get('contentDetails', {})
            bound_stream_id = content_details.get('boundStreamId')
            
            print(f"📊 Broadcast ID: {self.broadcast_id}")
            print(f"📊 Bound Stream ID: {bound_stream_id}")
            print(f"📊 Our Stream ID: {self.stream_id}")
            
            if bound_stream_id == self.stream_id:
                print("✅ Broadcast and stream are properly bound")
                return True
            else:
                print("❌ Broadcast and stream are not properly bound")
                return False
                
        except Exception as e:
            print(f"❌ Error checking broadcast-stream binding: {e}")
            return False