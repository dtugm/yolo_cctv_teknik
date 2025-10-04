#!/usr/bin/env python3
"""
Simple test script for RTSP streaming functionality.
Creates a test pattern and streams it via RTSP for testing purposes.
"""

import argparse
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add paths for src
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import RTSPStreamingConfig
from src.streaming.rtsp_streamer import RTSPStreamer


def create_test_pattern(width: int, height: int, frame_count: int) -> np.ndarray:
    """Create a test pattern frame."""
    # Create a colorful test pattern
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create moving colored rectangles
    for i in range(3):
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]  # BGR colors
        x = int((frame_count * 2 + i * 100) % (width - 100))
        y = int((frame_count * 1 + i * 50) % (height - 100))
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), color, -1)
    
    # Add frame counter text
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {timestamp}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add resolution info
    cv2.putText(frame, f"Resolution: {width}x{height}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def main():
    parser = argparse.ArgumentParser(
        description='Test RTSP Streaming with Test Pattern'
    )
    
    # RTSP streaming arguments
    parser.add_argument('--rtsp-host', type=str, default='0.0.0.0',
                       help='RTSP server host (default: 0.0.0.0)')
    parser.add_argument('--rtsp-port', type=int, default=8554,
                       help='RTSP server port (default: 8554)')
    parser.add_argument('--rtsp-mount', type=str, default='/test',
                       help='RTSP mount point (default: /test)')
    parser.add_argument('--rtsp-resolution', type=str, default='720p',
                       choices=['480p', '720p', '1080p'],
                       help='RTSP stream resolution (default: 720p)')
    parser.add_argument('--rtsp-fps', type=int, default=30,
                       help='RTSP stream frame rate (default: 30)')
    parser.add_argument('--rtsp-bitrate', type=int, default=2000,
                       help='RTSP stream video bitrate in kbps (default: 2000)')
    parser.add_argument('--rtsp-transport', type=str, default='tcp',
                       choices=['tcp', 'udp'],
                       help='RTSP transport protocol (default: tcp)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Create RTSP configuration
    rtsp_config = RTSPStreamingConfig(
        enabled=True,
        host=args.rtsp_host,
        port=args.rtsp_port,
        mount_point=args.rtsp_mount,
        resolution=args.rtsp_resolution,
        frame_rate=args.rtsp_fps,
        video_bitrate=args.rtsp_bitrate,
        rtsp_transport=args.rtsp_transport,
        max_stream_duration=args.duration
    )
    
    print("🧪 Starting RTSP Test Pattern Streaming...")
    print(f"   RTSP Host: {args.rtsp_host}")
    print(f"   RTSP Port: {args.rtsp_port}")
    print(f"   RTSP Mount: {args.rtsp_mount}")
    print(f"   RTSP Resolution: {args.rtsp_resolution}")
    print(f"   RTSP FPS: {args.rtsp_fps}")
    print(f"   RTSP Bitrate: {args.rtsp_bitrate}k")
    print(f"   RTSP Transport: {args.rtsp_transport}")
    print(f"   Duration: {args.duration} seconds")
    print()
    
    # Initialize RTSP streamer
    rtsp_streamer = RTSPStreamer(rtsp_config)
    
    # Start RTSP stream
    if not rtsp_streamer.start():
        print("❌ Failed to start RTSP stream. Exiting.")
        return 1
    
    # Wait for stream to initialize
    print("⏳ Waiting for RTSP stream to initialize...")
    time.sleep(3)
    
    # Verify FFmpeg is healthy
    if not rtsp_streamer.check_ffmpeg_health():
        print("❌ FFmpeg process is not healthy. Exiting.")
        return 1
    
    print("✅ RTSP stream initialized and ready for frames")
    print(f"🌍 Stream URL: {rtsp_streamer.get_stream_url()}")
    print("   You can now connect to this stream using:")
    print("   - OBS Studio: Add Media Source with RTSP URL")
    print("   - VLC: Open Network Stream")
    print("   - FFplay: ffplay rtsp://host:port/test")
    print("   - Any RTSP-compatible client")
    print()
    
    # Get resolution dimensions
    resolution_map = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }
    width, height = resolution_map.get(args.rtsp_resolution, (1280, 720))
    
    try:
        print("🚀 Starting test pattern generation and streaming...")
        print("   Press Ctrl+C to stop streaming")
        print()
        
        frame_count = 0
        start_time = time.time()
        target_frame_interval = 1.0 / args.rtsp_fps
        
        while True:
            current_time = time.time()
            
            # Check duration limit
            if current_time - start_time > args.duration:
                print(f"⏰ Test duration ({args.duration}s) reached, stopping...")
                break
            
            # Frame rate limiting
            if current_time - start_time < frame_count * target_frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            # Create test pattern frame
            frame = create_test_pattern(width, height, frame_count)
            
            # Send frame to RTSP streamer
            rtsp_streamer.add_frame(frame)
            
            frame_count += 1
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📹 Sent {frame_count} frames ({fps:.1f} FPS)")
        
        print("\n✅ Test pattern streaming completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
    finally:
        # Stop the RTSP stream
        rtsp_streamer.stop()
        
        # Print final statistics
        stats = rtsp_streamer.get_stats()
        print(f"\n📊 Final Stream Statistics:")
        print(f"   Total frames streamed: {stats['frame_count']}")
        if stats.get('duration'):
            print(f"   Stream duration: {stats['duration']:.1f} seconds")
            print(f"   Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"   Stream URL: {stats['stream_url']}")


if __name__ == "__main__":
    main()
