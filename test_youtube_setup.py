#!/usr/bin/env python3
"""
Test script to verify YouTube streaming setup without starting actual inference.
Tests authentication, API access, and dependencies.
"""

import sys
import os
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("🔍 Testing dependencies...")
    
    try:
        import cv2
        print("✅ OpenCV installed")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy installed")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    try:
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        print("✅ Google API libraries installed")
    except ImportError as e:
        print(f"❌ Google API libraries missing: {e}")
        print("   Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return False
    
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg installed")
        else:
            print("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not installed or not in PATH")
        print("   Install FFmpeg: https://ffmpeg.org/download.html")
        return False
    
    return True


def test_client_secrets():
    """Test if client secrets file exists."""
    print("\n🔍 Testing client secrets...")
    
    client_secrets_file = "client_secrets.json"
    if os.path.exists(client_secrets_file):
        print(f"✅ Client secrets file found: {client_secrets_file}")
        return True
    else:
        print(f"❌ Client secrets file not found: {client_secrets_file}")
        print("   Download from Google Cloud Console and place in project root")
        print("   See YOUTUBE_STREAMING_SETUP.md for detailed instructions")
        return False


def test_imports():
    """Test if our custom modules can be imported."""
    print("\n🔍 Testing module imports...")
    
    try:
        from src.config.settings import YouTubeStreamingConfig
        print("✅ YouTubeStreamingConfig imported")
    except ImportError as e:
        print(f"❌ Failed to import YouTubeStreamingConfig: {e}")
        return False
    
    try:
        from src.streaming.youtube_streamer import YouTubeStreamer
        print("✅ YouTubeStreamer imported")
    except ImportError as e:
        print(f"❌ Failed to import YouTubeStreamer: {e}")
        return False
    
    return True


def test_youtube_api_mock():
    """Test YouTube API authentication (mock mode)."""
    print("\n🔍 Testing YouTube API setup...")
    
    if not os.path.exists("client_secrets.json"):
        print("⚠️  Skipping API test - no client secrets file")
        return True
    
    try:
        from src.config.settings import YouTubeStreamingConfig
        from src.streaming.youtube_streamer import YouTubeStreamer
        
        # Create config with test mode
        config = YouTubeStreamingConfig(
            enabled=True,
            client_secrets_file="client_secrets.json"
        )
        
        # Try to create streamer (this will test file validation)
        streamer = YouTubeStreamer(config)
        print("✅ YouTubeStreamer created successfully")
        
        # Note: We don't actually authenticate here to avoid browser popup
        print("⚠️  Authentication test skipped (run actual script to test)")
        
        return True
        
    except Exception as e:
        print(f"❌ YouTube API setup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 YouTube Streaming Setup Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Client Secrets", test_client_secrets),
        ("Module Imports", test_imports),
        ("YouTube API Setup", test_youtube_api_mock),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! You're ready to stream to YouTube.")
        print("\nNext steps:")
        print("1. Run: python predict_youtube_stream.py --source your_video.mp4")
        print("2. Complete OAuth authentication in browser")
        print("3. Your stream will start automatically")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nSee YOUTUBE_STREAMING_SETUP.md for detailed setup instructions.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
