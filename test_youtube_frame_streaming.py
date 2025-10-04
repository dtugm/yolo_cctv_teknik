#!/usr/bin/env python3
"""
Test script for YouTube frame streaming to debug frame flow issues.
This script tests the frame streaming pipeline without running full YOLO inference.
"""

import sys
from pathlib import Path
import time
import numpy as np
import cv2

# Add paths for src
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import YouTubeStreamingConfig
from src.streaming.youtube_streamer import YouTubeStreamer


def create_test_frame(width: int, height: int, frame_number: int) -> np.ndarray:
    """Create a test frame with frame number displayed."""
    # Create a colored background that changes over time
    hue = (frame_number * 10) % 360
    hsv = np.full((height, width, 3), [hue, 255, 255], dtype=np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add frame number text
    text = f"Test Frame {frame_number}"
    font_scale = 2.0
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Add timestamp
    timestamp = f"Time: {time.strftime('%H:%M:%S')}"
    cv2.putText(frame, timestamp, (50, height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return frame


def test_youtube_streaming():
    """Test YouTube streaming with synthetic frames."""
    print("🧪 Testing YouTube Frame Streaming...")
    
    # Create YouTube configuration
    youtube_config = YouTubeStreamingConfig(
        enabled=True,
        client_secrets_file='client_secrets.json',
        broadcast_title='CCTV Teknik Frame Test Stream',
        broadcast_description='Testing frame streaming without YOLO inference',
        privacy_status='unlisted',
        resolution='720p',
        video_bitrate=2500,
        auto_start=True
    )
    
    # Initialize YouTube streamer
    youtube_streamer = YouTubeStreamer(youtube_config)
    
    # Start YouTube stream
    if not youtube_streamer.start():
        print("❌ Failed to start YouTube stream. Exiting.")
        return 1
    
    # Wait for stream to initialize
    print("⏳ Waiting for YouTube stream to initialize...")
    time.sleep(5)
    
    # Verify FFmpeg is healthy
    if not youtube_streamer.check_ffmpeg_health():
        print("❌ FFmpeg process is not healthy. Exiting.")
        return 1
    
    print("✅ YouTube stream initialized and ready for frames")
    print("🎬 Starting to send test frames...")
    print("   Press Ctrl+C to stop streaming")
    print()
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Create test frame
            width, height = 1280, 720  # 720p resolution
            test_frame = create_test_frame(width, height, frame_count)
            
            # Send frame to YouTube streamer
            youtube_streamer.add_frame(test_frame)
            
            frame_count += 1
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📹 Sent {frame_count} frames ({fps:.1f} FPS)")
                
                # Check streamer stats
                stats = youtube_streamer.get_stats()
                print(f"   Streamer frame count: {stats['frame_count']}")
                print(f"   Streamer running: {stats['running']}")
            
            # Control frame rate (30 FPS)
            time.sleep(1.0 / 30.0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
    finally:
        # Stop the YouTube stream
        youtube_streamer.stop()
        
        # Print final statistics
        stats = youtube_streamer.get_stats()
        print(f"\n📊 Final Stream Statistics:")
        print(f"   Total frames sent: {frame_count}")
        print(f"   Streamer frames processed: {stats['frame_count']}")
        if stats.get('duration'):
            print(f"   Stream duration: {stats['duration']:.1f} seconds")
            print(f"   Average FPS: {stats['fps']:.1f}")
        if stats.get('stream_url'):
            print(f"   Stream URL: {stats['stream_url']}")


if __name__ == "__main__":
    test_youtube_streaming()
