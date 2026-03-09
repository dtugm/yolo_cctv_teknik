#!/usr/bin/env python3
"""
Raw RTSP Stream Tester with HTTP Output
Tests original CCTV resolution without YOLO processing
"""

import cv2
import threading
from flask import Flask, Response, render_template_string
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
RTSP_URL = "rtsp://10.2.10.70:7447/rNVL3fqfqxknwEJt"  # ⚠️ CHANGE THIS
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5050
TARGET_FPS = 30

# Request specific resolution (optional - try your 2K resolution)
REQUEST_WIDTH = 2560   # Set to 0 to use camera default
REQUEST_HEIGHT = 1440  # Set to 0 to use camera default

# JPEG encoding quality (1-100, higher = better quality)
JPEG_QUALITY = 95

# ==================== GLOBAL VARIABLES ====================
latest_frame = None
frame_lock = threading.Lock()
stream_info = {
    'resolution': 'Unknown',
    'fps': 0,
    'status': 'Initializing...',
    'frame_count': 0
}

# ==================== FLASK APP ====================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Raw RTSP Stream Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1920px;
            margin: 0 auto;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .info-panel {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .info-item {
            display: inline-block;
            margin-right: 30px;
            font-size: 16px;
        }
        .info-label {
            color: #888;
            font-weight: bold;
        }
        .info-value {
            color: #4CAF50;
        }
        .video-container {
            text-align: center;
            background: #000;
            padding: 10px;
            border-radius: 8px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 2px solid #4CAF50;
        }
        .status-ok { color: #4CAF50; }
        .status-error { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Raw RTSP Stream - Resolution Test</h1>
        
        <div class="info-panel">
            <div class="info-item">
                <span class="info-label">Resolution:</span>
                <span class="info-value" id="resolution">{{ resolution }}</span>
            </div>
            <div class="info-item">
                <span class="info-label">FPS:</span>
                <span class="info-value" id="fps">{{ fps }}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Status:</span>
                <span class="info-value" id="status">{{ status }}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Frames:</span>
                <span class="info-value" id="frames">{{ frame_count }}</span>
            </div>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="RTSP Stream">
        </div>
        
        <p style="text-align: center; color: #888; margin-top: 20px;">
            RTSP Source: <code>{{ rtsp_url }}</code>
        </p>
    </div>
    
    <script>
        // Auto-refresh stats every 2 seconds
        setInterval(function() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('resolution').textContent = data.resolution;
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('frames').textContent = data.frame_count;
                });
        }, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page with video stream"""
    return render_template_string(
        HTML_TEMPLATE,
        resolution=stream_info['resolution'],
        fps=stream_info['fps'],
        status=stream_info['status'],
        frame_count=stream_info['frame_count'],
        rtsp_url=RTSP_URL
    )

@app.route('/stats')
def stats():
    """JSON endpoint for live stats"""
    return stream_info

@app.route('/video_feed')
def video_feed():
    """Video streaming route - returns MJPEG stream"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_frames():
    """
    Generator function for MJPEG streaming
    Yields frames in multipart format for browser display
    """
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
        
        # Encode frame to JPEG with high quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not ret:
            continue
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==================== RTSP CAPTURE THREAD ====================
def capture_rtsp_stream():
    """
    Captures frames from RTSP stream and updates global frame
    Runs in separate thread
    """
    global latest_frame, stream_info
    
    print(f"🔗 Connecting to RTSP: {RTSP_URL}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(RTSP_URL)
    
    # Set resolution if specified
    if REQUEST_WIDTH > 0 and REQUEST_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_HEIGHT)
        print(f"📐 Requested resolution: {REQUEST_WIDTH}x{REQUEST_HEIGHT}")
    
    # Optimize for RTSP streaming
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        stream_info['status'] = 'ERROR: Cannot connect to RTSP'
        print("❌ Failed to open RTSP stream")
        return
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    stream_info['resolution'] = f"{actual_width}x{actual_height}"
    stream_info['status'] = 'Connected'
    
    print(f"✅ Connected successfully!")
    print(f"📺 Actual Resolution: {actual_width}x{actual_height}")
    print(f"🎬 Camera FPS: {actual_fps if actual_fps > 0 else 'Unknown'}")
    print(f"🌐 Stream available at: http://localhost:{HTTP_PORT}")
    
    # FPS calculation variables
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            stream_info['status'] = 'ERROR: Lost connection'
            print("⚠️ Failed to read frame, reconnecting...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue
        
        # Update frame count
        frame_count += 1
        fps_frame_count += 1
        stream_info['frame_count'] = frame_count
        
        # Calculate FPS every second
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            stream_info['fps'] = f"{fps_frame_count / elapsed:.1f}"
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Add resolution overlay on frame
        overlay_text = f"Resolution: {actual_width}x{actual_height} | Frame: {frame_count}"
        cv2.putText(
            frame, overlay_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp, (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # Update global frame
        with frame_lock:
            latest_frame = frame
        
        # Control frame rate
        time.sleep(1.0 / TARGET_FPS)
    
    cap.release()

# ==================== MAIN EXECUTION ====================
def main():
    """Main function to start capture thread and Flask server"""
    
    print("=" * 60)
    print("🎥 RAW RTSP STREAM TESTER")
    print("=" * 60)
    print(f"RTSP URL: {RTSP_URL}")
    print(f"HTTP Server: http://{HTTP_HOST}:{HTTP_PORT}")
    print(f"JPEG Quality: {JPEG_QUALITY}%")
    print("=" * 60)
    
    # Start RTSP capture in separate thread
    capture_thread = threading.Thread(target=capture_rtsp_stream, daemon=True)
    capture_thread.start()
    
    # Wait a moment for connection
    time.sleep(2)
    
    # Start Flask server
    print("\n🚀 Starting HTTP server...")
    print(f"📱 Open browser: http://localhost:{HTTP_PORT}")
    print("Press Ctrl+C to stop\n")
    
    try:
        app.run(
            host=HTTP_HOST,
            port=HTTP_PORT,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")

if __name__ == "__main__":
    main()
