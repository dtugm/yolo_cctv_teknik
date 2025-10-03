#!/usr/bin/env python3
"""
YouTube Live streaming prediction script.
Runs YOLO inference and broadcasts results directly to YouTube Live.
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
    VisualizationConfig, YouTubeStreamingConfig
)
from src.streaming.youtube_streamer import YouTubeStreamer


class YouTubeStreamInference(InferenceEngine):
    """Extended inference engine with YouTube Live streaming support."""
    
    def __init__(self, youtube_streamer: YouTubeStreamer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.youtube_streamer = youtube_streamer
    
    def write_results(self, idx: int, preds, batch):
        """Override to push frames to YouTube streaming server."""
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
                print(f"   YouTube streamer running: {self.youtube_streamer and self.youtube_streamer.is_running()}")
            
            # Push to YouTube streamer
            if self.youtube_streamer and self.youtube_streamer.is_running():
                try:
                    self.youtube_streamer.add_frame(annotated_frame)
                    if self._frame_debug_count % 30 == 0:
                        print(f"✅ Frame sent to YouTube streamer")
                except Exception as e:
                    print(f"Error pushing frame to YouTube stream: {e}")
            else:
                if self._frame_debug_count % 30 == 0:
                    print("⚠️  YouTube streamer not available or not running")
                    if self.youtube_streamer:
                        print(f"   Streamer running: {self.youtube_streamer.is_running()}")
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
    
    print("🔴 Starting YOLO YouTube Live Streaming...")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Confidence: {args.conf}")
    print(f"   YouTube Title: {args.youtube_title}")
    print(f"   YouTube Privacy: {args.youtube_privacy}")
    print(f"   YouTube Resolution: {args.youtube_resolution}")
    if args.manual_transition:
        print(f"   Transition Mode: Manual")
    print()
    
    # Initialize YouTube streamer
    youtube_streamer = YouTubeStreamer(youtube_config)
    
    # Start YouTube stream
    if not youtube_streamer.start():
        print("❌ Failed to start YouTube stream. Exiting.")
        return 1
    
    # Wait for stream to initialize and FFmpeg to be ready
    print("⏳ Waiting for YouTube stream to initialize...")
    time.sleep(5)  # Increased wait time for FFmpeg to be fully ready
    
    # Verify FFmpeg is healthy before starting inference
    if not youtube_streamer.check_ffmpeg_health():
        print("❌ FFmpeg process is not healthy. Exiting.")
        return 1
    
    print("✅ YouTube stream initialized and ready for frames")
    
    # Initialize inference engine with YouTube streaming
    engine = YouTubeStreamInference(
        youtube_streamer=youtube_streamer,
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )
    
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
        # Always stop the YouTube stream
        youtube_streamer.stop()
        
        # Print final statistics
        stats = youtube_streamer.get_stats()
        print(f"\n📊 Final Stream Statistics:")
        print(f"   Total frames streamed: {stats['frame_count']}")
        if stats.get('duration'):
            print(f"   Stream duration: {stats['duration']:.1f} seconds")
            print(f"   Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"   Stream URL: {stats['stream_url']}")


if __name__ == "__main__":
    main()