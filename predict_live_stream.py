#!/usr/bin/env python3
"""
Live streaming prediction script with HTTP server.
Runs YOLO inference and broadcasts results via HTTP MJPEG stream.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time

# Add paths for src and deep_sort_pytorch
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics" / "yolo" / "v8" / "detect"))

from src.core.inference import InferenceEngine
from src.config.settings import (
    InferenceConfig, TrackingConfig, 
    VisualizationConfig, StreamingConfig
)
from src.streaming.server import StreamingServer
from src.capture.plate_capture import PlateCaptureManager
from src.config.settings import PlateCaptureConfig

class LiveStreamInference(InferenceEngine):
    """Extended inference engine with HTTP streaming and plate capture support."""
    
    def __init__(
        self,
        streaming_server: StreamingServer,
        plate_capture_manager: PlateCaptureManager = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.streaming_server = streaming_server
        self.plate_capture_manager = plate_capture_manager
    
    def write_results(self, idx: int, preds, batch):
        """Override to push frames to streaming server and capture violations."""
        log_string = super().write_results(idx, preds, batch)
        
        # Get the annotated frame from the annotator
        if hasattr(self, 'annotator') and self.annotator is not None:
            annotated_frame = self.annotator.result()
            
            # Push to streaming server
            if self.streaming_server and self.streaming_server.running:
                try:
                    self.streaming_server.add_frame(annotated_frame)
                except Exception as e:
                    print(f"Error pushing frame to stream: {e}")
            
            # Check for speed violations and capture plates
            if self.plate_capture_manager and self.plate_capture_manager.enabled:
                self._check_speed_violations(batch)
        
        return log_string
    
    def _check_speed_violations(self, batch):
        """Check tracked objects for speed violations and capture."""
        if not self.plate_capture_manager or not self.plate_capture_manager.enabled:
            return
        
        # Check tracker exists
        if not hasattr(self, 'tracker') or self.tracker is None:
            return
        
        try:
            # Extract original frame from batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                im0s = batch[2]
            else:
                return
            
            # Handle list of images
            frame = im0s[0] if isinstance(im0s, list) else im0s
            
            # Validate frame
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return
            
            # Check if tracker has tracks
            if not hasattr(self.tracker, 'tracks') or len(self.tracker.tracks) == 0:
                return
            
            # Process each tracked object
            for track_id, track in self.tracker.tracks.items():
                try:
                    # Get speed
                    speed = track.speed
                    
                    if speed is None or speed <= 0:
                        continue
                    
                    # Check if confirmed track
                    if not track.is_confirmed():
                        continue
                    
                    # Get bounding box
                    bbox = track.to_tlbr()
                    
                    # Get class name
                    class_name = f"class_{track.class_id}"
                    
                    # Check and capture violation
                    captured_path = self.plate_capture_manager.check_and_capture(
                        frame=frame,
                        track_id=track_id,
                        bbox=tuple(map(int, bbox)),
                        speed=speed,
                        class_name=class_name
                    )
                    
                    if captured_path:
                        print(f"✅ Captured violation: Track {track_id} @ {speed:.1f} km/h")
                    
                except Exception as track_error:
                    print(f"⚠️  Error processing track {track_id}: {track_error}")
                    continue
        
        except Exception as e:
            print(f"❌ Error in speed violation check: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='YOLO Object Detection with Live HTTP Streaming'
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
    parser.add_argument('--port', type=int, default=5050,
                       help='HTTP streaming port (default: 5050)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='HTTP server host (default: 0.0.0.0)')
    parser.add_argument('--show', action='store_true',
                       help='Also display results in window')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--counter-direction', type=str, default='north_enter', choices=['north_enter', 'south_enter'],
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
    parser.add_argument('--enable-plate-capture', action='store_true', default=True,
                       help='Enable plate capture for speed violations (default: True)')
    parser.add_argument('--disable-plate-capture', action='store_true',
                       help='Disable plate capture')
    parser.add_argument('--speed-limit', type=float, default=60.0,
                       help='Speed limit in km/h for violations (default: 60.0)')
    parser.add_argument('--violation-output-dir', type=str, default='output/violations',
                       help='Output directory for violation captures (default: output/violations)')
    parser.add_argument('--capture-quality', type=int, default=95,
                       help='JPEG quality for captured images (default: 95)')
    parser.add_argument('--pixels-per-meter', type=float, default=20.0, 
                       help='Calibration: pixels per meter for speed calculation')
    parser.add_argument('--fps', type=float, default=30.0, 
                       help='Video frame rate for speed calculation')

    
    args = parser.parse_args()
    
    # Handle visibility and feature logic
    show_info_panel = args.show_info_panel and not args.hide_info_panel
    show_counters = args.show_counters and not args.hide_counters
    enable_keyboard_reset = args.enable_keyboard_reset and not args.disable_keyboard_reset
    enable_auto_reset = args.enable_auto_reset and not args.disable_auto_reset

    # Handle plate capture logic
    enable_plate_capture = args.enable_plate_capture and not args.disable_plate_capture
    
    # Create plate capture config
    plate_capture_config = PlateCaptureConfig(
        enabled=enable_plate_capture,
        output_dir=args.violation_output_dir,
        speed_limit=args.speed_limit,
        image_quality=args.capture_quality
    )
    
    # Create configurations
    inference_config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    tracking_config = TrackingConfig()
    visualization_config = VisualizationConfig(
        counter_direction=args.counter_direction,
        show_info_panel=show_info_panel,
        show_counters=show_counters,
        enable_keyboard_reset=enable_keyboard_reset,
        enable_auto_daily_reset=enable_auto_reset
    )
    
    streaming_config = StreamingConfig(
        enabled=True,
        host=args.host,
        port=args.port
    )
    
    # Hydra config for BasePredictor
    # Ensure imgsz is a list for compatibility
    imgsz = [640, 640]  # Default image size as list
    
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
    
    print(f"🚀 Starting YOLO Live Streaming Inference...")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Device: {args.device}")
    print(f"   Stream: http://{args.host}:{args.port}/_yolo_stream/")
    print()
    
    # Initialize streaming server
    streaming_server = StreamingServer(streaming_config)
    streaming_server.start()
    
    # Wait a moment for server to initialize
    time.sleep(1)
    
    # Initialize inference engine with streaming
    engine = LiveStreamInference(
        streaming_server=streaming_server,
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )

    # Initialize plate capture manager
    plate_capture_manager = PlateCaptureManager(
        output_dir=plate_capture_config.output_dir,
        speed_limit=plate_capture_config.speed_limit,
        enabled=plate_capture_config.enabled,
        save_metadata=plate_capture_config.save_metadata,
        image_format=plate_capture_config.image_format,
        image_quality=plate_capture_config.image_quality
    )
    
    # Initialize inference engine with plate capture
    engine = LiveStreamInference(
        streaming_server=streaming_server,
        plate_capture_manager=plate_capture_manager,  # Pass to engine
        inference_config=inference_config,
        tracking_config=tracking_config,
        visualization_config=visualization_config,
        hydra_config=hydra_config
    )
    
    try:
        results = engine()
        print("\n✅ Inference completed!")
        
        # Print plate capture statistics
        if enable_plate_capture:
            plate_capture_manager.print_statistics()
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        if enable_plate_capture:
            plate_capture_manager.print_statistics()
    finally:
        streaming_server.stop()


if __name__ == "__main__":
    main()

