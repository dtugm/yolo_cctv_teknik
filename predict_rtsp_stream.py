#!/usr/bin/env python3
"""
HTTP streaming prediction script.
Runs YOLO inference and serves results as HTTP stream using FFmpeg.
"""

import argparse
import sys
from pathlib import Path
import time

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import (
    InferenceConfig, TrackingConfig, 
    VisualizationConfig, RTSPStreamingConfig
)
from src.streaming.rtsp_streamer import RTSPStreamer


class HTTPStreamInference(InferenceEngine):
    """Extended inference engine with HTTP streaming support."""
    
    def __init__(self, streamer: RTSPStreamer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streamer = streamer
    
    def write_results(self, idx: int, preds, batch):
        """Override to push frames to HTTP streaming server."""
        log_string = super().write_results(idx, preds, batch)
        
        # Debug: Print frame info every 30 frames
        if not hasattr(self, '_frame_debug_count'):
            self._frame_debug_count = 0
        self._frame_debug_count += 1
        
        # Get the annotated frame from the inference engine
        # Check multiple possible locations for the annotated frame
        annotated_frame = None
        
        # First try: check if annotated_frame attribute exists
        if hasattr(self, 'annotated_frame') and self.annotated_frame is not None:
            annotated_frame = self.annotated_frame
        
        # Second try: check if annotator exists and has the frame
        elif hasattr(self, 'annotator') and self.annotator is not None:
            annotated_frame = self.annotator.result()
        
        # Third try: get frame from batch data
        elif batch and len(batch) >= 3:
            _, _, im0 = batch
            if im0 is not None:
                annotated_frame = im0.copy()
        
        if annotated_frame is not None:
            if self._frame_debug_count % 30 == 0:
                print(f"🎬 Inference engine processed frame {self._frame_debug_count}")
                print(f"   Frame shape: {annotated_frame.shape}")
                print(f"   Frame dtype: {annotated_frame.dtype}")
                print(f"   HTTP streamer running: {self.streamer and self.streamer.is_running()}")
            
            # Push to HTTP streamer
            if self.streamer and self.streamer.is_running():
                try:
                    self.streamer.add_frame(annotated_frame)
                    if self._frame_debug_count % 30 == 0:
                        print(f"✅ Frame sent to HTTP streamer")
                except Exception as e:
                    print(f"Error pushing frame to HTTP stream: {e}")
            else:
                if self._frame_debug_count % 30 == 0:
                    print("⚠️  HTTP streamer not available or not running")
                    if self.streamer:
                        print(f"   Streamer running: {self.streamer.is_running()}")
                    else:
                        print("   Streamer is None")
        else:
            if self._frame_debug_count % 30 == 0:
                print(f"⚠️  No annotated frame available for frame {self._frame_debug_count}")
                print(f"   Has annotated_frame attr: {hasattr(self, 'annotated_frame')}")
                print(f"   Has annotator attr: {hasattr(self, 'annotator')}")
                print(f"   Batch length: {len(batch) if batch else 'None'}")
        
        return log_string


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection with HTTP Streaming'
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
    
    # HTTP streaming specific arguments
    parser.add_argument('--http-host', type=str, default='0.0.0.0',
                       help='HTTP server host (default: 0.0.0.0)')
    parser.add_argument('--http-port', type=int, default=8554,
                       help='HTTP server port (default: 8554)')
    parser.add_argument('--http-mount', type=str, default='/live',
                       help='HTTP mount point (default: /live)')
    parser.add_argument('--stream-resolution', type=str, default='720p',
                       choices=['480p', '720p', '1080p'],
                       help='Stream resolution (default: 720p)')
    parser.add_argument('--stream-fps', type=int, default=30,
                       help='Stream frame rate (default: 30)')
    parser.add_argument('--stream-bitrate', type=int, default=2000,
                       help='Stream video bitrate in kbps (default: 2000)')
    
    args = parser.parse_args()
    
    # Create configurations
    inference_config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        image_size=args.imgsz
    )
    
    tracking_config = TrackingConfig()
    visualization_config = VisualizationConfig()
    
    stream_config = RTSPStreamingConfig(
        enabled=True,
        host=args.http_host,
        port=args.http_port,
        mount_point=args.http_mount,
        resolution=args.stream_resolution,
        frame_rate=args.stream_fps,
        video_bitrate=args.stream_bitrate
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
    }
    
    print("🔴 Starting YOLO HTTP Streaming...")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Confidence: {args.conf}")
    print(f"   HTTP Host: {args.http_host}")
    print(f"   HTTP Port: {args.http_port}")
    print(f"   HTTP Mount: {args.http_mount}")
    print(f"   Stream Resolution: {args.stream_resolution}")
    print(f"   Stream FPS: {args.stream_fps}")
    print(f"   Stream Bitrate: {args.stream_bitrate}k")
    print()
    
    # Initialize HTTP streamer
    streamer = RTSPStreamer(stream_config)
    
    # Start HTTP stream
    if not streamer.start():
        print("❌ Failed to start HTTP stream. Exiting.")
        return 1
    
    # Wait for stream to initialize and FFmpeg to be ready
    print("⏳ Waiting for HTTP stream to initialize...")
    time.sleep(5)  # Wait time for FFmpeg to be fully ready
    
    # Verify FFmpeg is healthy before starting inference
    if not streamer.check_ffmpeg_health():
        print("❌ FFmpeg process is not healthy. Exiting.")
        return 1
    
    print("✅ HTTP stream initialized and ready for frames")
    
    # Initialize inference engine with HTTP streaming
    engine = HTTPStreamInference(
        streamer=streamer,
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )
    
    try:
        print("🚀 Starting inference and streaming to HTTP...")
        print("   Press Ctrl+C to stop streaming")
        print(f"   Stream URL: {streamer.get_stream_url()}")
        print("   Connect to this stream using:")
        print("   - OBS Studio: Add Media Source with HTTP URL")
        print("   - VLC: Open Network Stream")
        print("   - FFplay: ffplay http://host:port/live")
        print("   - Any HTTP-compatible client")
        print()
        
        # Run prediction
        results = engine()
        print("\n✅ Inference completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
    finally:
        # Always stop the HTTP stream
        streamer.stop()
        
        # Print final statistics
        stats = streamer.get_stats()
        print(f"\n📊 Final Stream Statistics:")
        print(f"   Total frames streamed: {stats['frame_count']}")
        if stats.get('duration'):
            print(f"   Stream duration: {stats['duration']:.1f} seconds")
            print(f"   Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"   Stream URL: {stats['stream_url']}")


if __name__ == "__main__":
    main()
