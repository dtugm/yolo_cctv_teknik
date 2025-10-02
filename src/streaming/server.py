"""HTTP streaming server for broadcasting annotated video frames."""

import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, make_response
from typing import Optional
import socket

from src.config.settings import StreamingConfig


class StreamingServer:
    """
    HTTP MJPEG streaming server for live annotated frames.
    Designed to be easily separable into a standalone service.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming server.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or StreamingConfig()
        self.app = Flask(__name__)
        self.running = False
        
        # Thread-safe frame storage
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self.last_update = 0.0
        self.frame_count = 0
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for streaming."""
        prefix = self.config.url_prefix.rstrip('/')
        
        @self.app.route(f'{prefix}/')
        def index():
            """Home page with embedded video player."""
            return f'''
            <!doctype html>
            <html>
            <head>
                <title>YOLO Live Stream</title>
                <style>
                    body {{
                        background: #000;
                        color: #fff;
                        text-align: center;
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                    }}
                    h2 {{ margin: 20px 0; }}
                    img {{
                        max-width: 95%;
                        height: auto;
                        border-radius: 6px;
                        border: 1px solid #333;
                    }}
                    .info {{
                        font-size: 12px;
                        color: #bbb;
                        margin-top: 20px;
                    }}
                    a {{
                        color: #4CAF50;
                        text-decoration: none;
                    }}
                    a:hover {{ text-decoration: underline; }}
                </style>
            </head>
            <body>
                <h2>🚗 YOLO Live Detection Stream</h2>
                <p>Snapshot: <a href="{prefix}/snapshot.jpg" target="_blank">{prefix}/snapshot.jpg</a></p>
                <img src="{prefix}/video_feed" alt="Live Stream"/>
                <p class="info">
                    If the image is blank, open <code>{prefix}/snapshot.jpg</code> to test a single frame.
                </p>
            </body>
            </html>
            '''
        
        @self.app.route(f'{prefix}/video_feed')
        def video_feed():
            """Video streaming route (MJPEG)."""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route(f'{prefix}/snapshot.jpg')
        def snapshot():
            """Single frame snapshot route."""
            with self.lock:
                if self.latest_jpeg:
                    resp = make_response(self.latest_jpeg)
                    resp.headers['Content-Type'] = 'image/jpeg'
                    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                    return resp
            
            # Return black image if no frame available
            black = np.zeros((16, 16, 3), dtype='uint8')
            ret, buf = cv2.imencode('.jpg', black)
            resp = make_response(buf.tobytes())
            resp.headers['Content-Type'] = 'image/jpeg'
            return resp
    
    def _generate_frames(self):
        """Generator for MJPEG stream."""
        while self.running:
            with self.lock:
                data = self.latest_jpeg
            
            if data:
                try:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
                    )
                except GeneratorExit:
                    return
                except Exception as e:
                    print(f"[StreamingServer] Yield error: {e}")
            else:
                time.sleep(0.05)
                continue
            
            time.sleep(0.01)  # Light throttle
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a new frame to the stream.
        
        Args:
            frame: BGR numpy array (cv2 image)
        """
        if frame is None:
            return
        
        try:
            # Encode frame to JPEG
            ret, buf = cv2.imencode(
                '.jpg', frame, 
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            )
            
            if not ret:
                print("[StreamingServer] Frame encoding failed")
                return
            
            jpg = buf.tobytes()
            
            # Update latest frame
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_jpeg = jpg
                self.last_update = time.time()
                self.frame_count += 1
            
            # Debug log (every 30 frames)
            if self.frame_count % 30 == 0:
                print(f"[StreamingServer] Frames: {self.frame_count}, last_update={self.last_update:.2f}")
                
        except Exception as e:
            print(f"[StreamingServer] add_frame error: {e}")
    
    def start(self) -> bool:
        """
        Start the streaming server in a background thread.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if not self.config.enabled:
            print("[StreamingServer] Streaming is disabled in config")
            return False
        
        # Check if port is available
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            res = sock.connect_ex(('127.0.0.1', self.config.port))
            sock.close()
            if res == 0:
                print(f"[StreamingServer] Warning: port {self.config.port} may be in use")
        except Exception as e:
            print(f"[StreamingServer] Port check error: {e}")
        
        def run_app():
            try:
                self.running = True
                self.app.run(
                    host=self.config.host,
                    port=self.config.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e:
                print(f"[StreamingServer] Server error: {e}")
            finally:
                self.running = False
        
        # Start server in daemon thread
        thread = threading.Thread(target=run_app, daemon=True)
        thread.start()
        
        # Wait for server to start
        time.sleep(0.8)
        
        prefix = self.config.url_prefix
        print(f"🌐 HTTP stream started at: http://localhost:{self.config.port}{prefix}/")
        print(f"   Video feed -> http://localhost:{self.config.port}{prefix}/video_feed")
        print(f"   Snapshot   -> http://localhost:{self.config.port}{prefix}/snapshot.jpg")
        
        return True
    
    def stop(self) -> None:
        """Stop the streaming server."""
        self.running = False
        print("[StreamingServer] Server stopped")

