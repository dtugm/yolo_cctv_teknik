# stream_server.py
import cv2
import threading
import time
from flask import Flask, render_template, Response
import queue
import numpy as np
from pathlib import Path
import subprocess
import sys
import os

app = Flask(__name__)

# Global variables for frame sharing
frame_queue = queue.Queue(maxsize=2)
latest_frame = None
frame_lock = threading.Lock()

# Initialize YOLO model in main thread
print("🔄 Initializing YOLO model in main thread...")
yolo_model = None
deep_sort_tracker = None

def initialize_models():
    """Initialize YOLO and DeepSort models in main thread"""
    global yolo_model, deep_sort_tracker
    
    try:
        # Initialize YOLO
        yolo_model = YOLO('yolov8n.pt')  # or your model path
        print("✅ YOLO model initialized in main thread")
        
        # Initialize DeepSort
        deep_sort_tracker = DeepSort(
            model_path='deep_sort/deep/checkpoint/ckpt.t7',  # your model path
            max_dist=0.2,
            min_confidence=0.3,
            nms_max_overlap=1.0,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
        )
        print("✅ DeepSort tracker initialized in main thread")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return False

class YOLOStreamer:
    def __init__(self, source="rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV", model="yolov8n.pt"):
        self.source = source
        self.model = model
        self.cap = None
        self.running = False
        self.detection_process = None
        
    def start_yolo_detection(self):
        """Initialize YOLO model and tracker"""
        try:
            print("🔄 Initializing YOLO model...")
            
            # Initialize YOLO model without signal handling
            import signal
            import os
            
            # Temporarily disable signal handling
            old_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
            
            self.model = YOLO(self.yolo_model)
            
            # Restore signal handler
            signal.signal(signal.SIGINT, old_sigint_handler)
            
            print("✅ YOLO model initialized successfully")
            
            # Initialize tracker
            self.tracker = DeepSort(
                model_path=self.deep_sort_model,
                max_dist=0.2,
                min_confidence=0.3,
                nms_max_overlap=1.0,
                max_iou_distance=0.7,
                max_age=70,
                n_init=3,
                nn_budget=100,
            )
            
            print("✅ DeepSort tracker initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing YOLO: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def generate_frames(self):
        """Generate frames with YOLO detection"""
        global latest_frame
        
        print(f"🚀 Starting YOLOStreamer with source: {self.source}")
        print(f"🔍 Source type: {type(self.source)}")
        
        if not self.start_yolo_detection():
            print("❌ Failed to start YOLO detection")
            return
            
        try:
            print(f"🔄 Attempting to connect to: {self.source}")
            
            # Initialize video capture
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                print("📹 Using webcam source")
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                print(f"🌐 Using network source: {self.source}")
                self.cap = cv2.VideoCapture(self.source)
                
                # RTSP specific settings
                if str(self.source).startswith('rtsp://'):
                    print("🔧 Applying RTSP optimizations...")
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 25)
                    
            # Test connection
            if not self.cap.isOpened():
                print(f"❌ Could not open video source: {self.source}")
                return
            else:
                print("✅ Video capture opened successfully")
                
            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret:
                print("❌ Could not read test frame")
                return
            else:
                print(f"✅ Successfully read test frame: {test_frame.shape}")
            
            self.running = True
            frame_count = 0
            
            print("🎬 Starting frame processing loop...")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"❌ Failed to read frame {frame_count}")
                    time.sleep(0.1)
                    continue
                    
                try:
                    # Process frame
                    processed_frame = self.process_frame_with_yolo(frame)
                    
                    with frame_lock:
                        latest_frame = processed_frame.copy()
                    
                    # Add to queue
                    try:
                        frame_queue.put(processed_frame, block=False)
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put(processed_frame, block=False)
                        except queue.Empty:
                            pass
                    
                    frame_count += 1
                    if frame_count == 1:
                        print("✅ First frame processed successfully!")
                    elif frame_count % 100 == 0:  # Log every 100 frames
                        print(f"✅ Processed {frame_count} frames")
                    
                except Exception as e:
                    print(f"❌ Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                
                time.sleep(0.033)  # ~30 FPS
                    
        except Exception as e:
            print(f"❌ Error in frame generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                print("🔄 Video capture released")


    
    def process_frame_with_yolo(self, frame):
        """Process a single frame through YOLO detection"""
        try:
            # Create a temporary dataset-like object for single frame
            class SingleFrameDataset:
                def __init__(self, frame):
                    self.frame = frame
                    self.mode = 'image'
                    self.count = 1
            
            # Prepare the frame for YOLO processing
            im0 = frame.copy()
            
            # Run inference
            results = self.predictor.model(im0)
            
            # Process results
            if results and len(results) > 0:
                # Get predictions
                pred = results[0].boxes
                
                if pred is not None and len(pred) > 0:
                    # Convert to the format expected by write_results
                    det = pred.data  # xyxy, conf, cls
                    
                    # Create batch-like structure
                    batch = (Path("stream"), frame[None], im0)
                    preds = [det]
                    
                    # Set up predictor attributes
                    self.predictor.seen = 0
                    self.predictor.dataset = SingleFrameDataset(frame)
                    self.predictor.save_dir = Path("runs/detect/stream")
                    self.predictor.webcam = False
                    
                    # Process the results
                    self.predictor.write_results(0, preds, batch)
                    
                    return im0
            
            # If no detections, still draw UI elements
            from live_pred import draw_boxes
            draw_boxes(im0, [], self.predictor.model.names, [], [])
            return im0
            
        except Exception as e:
            print(f"Error in YOLO processing: {e}")
            return frame
    
    def stop(self):
        """Stop the streaming"""
        self.running = False
        if self.cap:
            self.cap.release()

# Global streamer instance
streamer = None

def generate_video_stream():
    """Generator function for video streaming"""
    while True:
        try:
            # Get frame from queue or use latest frame
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                with frame_lock:
                    if latest_frame is not None:
                        frame = latest_frame.copy()
                    else:
                        # Create a black frame with text
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, 'Waiting for video...', (200, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in video stream generation: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Video streaming home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

from urllib.parse import unquote

@app.route('/start/<path:source>')
def start_stream(source):
    """Start streaming with specified source"""
    global streamer
    
    print(f"Received source parameter: {source}")
    
    if streamer and streamer.running:
        streamer.stop()
        time.sleep(1)
    
    # Handle different source types
    if source == 'cam':
        actual_source = 0
    elif source.startswith('rtsp://') or source.startswith('rtsp%3A//'):
        # Decode URL if it's encoded
        actual_source = unquote(source)
        print(f"Decoded RTSP URL: {actual_source}")
    else:
        actual_source = source
    
    print(f"Starting stream with source: {actual_source}")
    
    try:
        streamer = YOLOStreamer(source=actual_source)
        
        # Start streaming in a separate thread
        stream_thread = threading.Thread(target=streamer.generate_frames)
        stream_thread.daemon = True
        stream_thread.start()
        
        return f"Started streaming from {actual_source}"
    except Exception as e:
        print(f"Error starting stream: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"


@app.route('/stop')
def stop_stream():
    """Stop streaming"""
    global streamer
    if streamer:
        streamer.stop()
        streamer = None
    return "Streaming stopped"

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Live Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        .video-stream {
            max-width: 100%;
            height: auto;
            border: 2px solid #333;
            border-radius: 5px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            padding: 10px 20px;
            margin: 5px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .btn-danger {
            background-color: #dc3545;
        }
        .btn-danger:hover {
            background-color: #c82333;
        }
        .info {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 YOLO Live Vehicle Detection Stream</h1>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Live Stream">
        </div>
        
        <div class="controls">
            <a href="/start/cam" class="btn">📹 Start Webcam</a>
            <button onclick="startRTSP()" class="btn">📡 Start RTSP Stream</button>
            <a href="/stop" class="btn btn-danger">⏹️ Stop Stream</a>
        </div>

        <script>
        function startRTSP() {
            const rtspUrl = "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV";
            console.log('Starting RTSP stream with URL:', rtspUrl);
            
            // Use fetch to start the stream
            fetch('/start/' + encodeURIComponent(rtspUrl))
                .then(response => response.text())
                .then(data => {
                    console.log('Response:', data);
                    // Wait a moment for the stream to initialize, then reload
                    setTimeout(() => {
                        location.reload();
                    }, 2000);
                })
                .catch(error => {
                    console.error('Error starting RTSP stream:', error);
                    alert('Error starting RTSP stream: ' + error);
                });
        }
        </script>

        <div class="info">
            <h3>📋 Instructions:</h3>
            <ul>
                <li><strong>Webcam:</strong> Click "Start Webcam" to use your default camera</li>
                <li><strong>RTSP Stream:</strong> Modify the RTSP URL in the button above or use the custom URL format</li>
                <li><strong>Custom Source:</strong> Use /start/YOUR_SOURCE_HERE in the URL</li>
                <li><strong>Stop:</strong> Click "Stop Stream" to end the current stream</li>
            </ul>
            
            <h3>🎯 Features:</h3>
            <ul>
                <li>Real-time vehicle detection and tracking</li>
                <li>Speed estimation</li>
                <li>Vehicle counting (entering/leaving)</li>
                <li>Live FPS and system statistics</li>
                <li>Support for webcam and RTSP streams</li>
            </ul>
            
            <h3>🔧 Custom RTSP URL Format:</h3>
            <p>Replace underscores with appropriate characters:</p>
            <code>/start/rtsp_YOUR_IP_PORT_PATH</code>
            <p>Example: <code>/start/rtsp_192.168.1.100_554_stream1</code></p>
        </div>
    </div>
    
    <script>
        // Auto-refresh the page if the image fails to load
        document.querySelector('.video-stream').onerror = function() {
            setTimeout(function() {
                location.reload();
            }, 5000);
        };
    </script>
</body>
</html>
    '''
    
    # Write the HTML template
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("🚀 Starting YOLO Live Stream Server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎥 Make sure your live-pred.py is in the same directory")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
