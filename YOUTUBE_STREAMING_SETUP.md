# YouTube Live Streaming Setup Guide

This guide explains how to set up YouTube Live streaming for your YOLO inference results.

## Prerequisites

### 1. YouTube Channel Requirements

- **Verified YouTube Channel**: Your channel must be verified with a phone number
- **Live Streaming Enabled**: Enable live streaming in YouTube Studio (may take up to 24 hours)
- **No Recent Strikes**: Ensure no community guideline strikes in the last 90 days

### 2. System Requirements

- **FFmpeg**: Must be installed on your system
  - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [FFmpeg official site](https://ffmpeg.org/download.html)

### 3. Python Dependencies

Install the YouTube streaming requirements:

```bash
pip install -r requirements_youtube.txt
```

## YouTube API Setup

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.developers.google.com/)
2. Create a new project or select an existing one
3. Enable the **YouTube Data API v3**:
   - Go to "APIs & Services" > "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"

### 2. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Configure the consent screen if prompted:
   - Choose "External" for user type
   - Fill in required fields (app name, user support email, developer email)
   - Add your email to test users during development
4. For Application type, choose "Desktop application"
5. Download the JSON file and save it as `client_secrets.json` in your project root

### 3. Authentication Flow

On first run, the script will:

1. Open a browser window for Google OAuth
2. Ask you to sign in to your Google account
3. Request permission to manage your YouTube channel
4. Save authentication tokens for future use

## Usage

### Basic Usage

```bash
python predict_youtube_stream.py --source path/to/video.mp4
```

### RTSP Camera Stream

```bash
python predict_youtube_stream.py --source "rtsp://camera-ip:port/stream"
```

### Full Example with Options

```bash
python predict_youtube_stream.py \
    --source datasets/test.mp4 \
    --model yolov8n.pt \
    --conf 0.25 \
    --youtube-title "My YOLO Detection Stream" \
    --youtube-description "Real-time object detection demo" \
    --youtube-privacy public \
    --youtube-resolution 720p \
    --youtube-bitrate 2500 \
    --show
```

## Configuration Options

### Inference Options

- `--source`: Video file or RTSP stream URL (required)
- `--model`: YOLO model path (default: yolov8n.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.7)
- `--device`: Device to run on (default: cuda)
- `--imgsz`: Inference image size (default: 640)

### YouTube Options

- `--youtube-title`: Broadcast title
- `--youtube-description`: Broadcast description
- `--youtube-privacy`: public/private/unlisted (default: public)
- `--youtube-resolution`: 480p/720p/1080p (default: 720p)
- `--youtube-bitrate`: Video bitrate in kbps (default: 2500)
- `--client-secrets`: Path to client secrets file (default: client_secrets.json)

### Display Options

- `--show`: Also display results in local window
- `--save`: Save results to file

## Stream Quality Settings

### Resolution and Bitrate Recommendations

| Resolution | Recommended Bitrate | Use Case         |
| ---------- | ------------------- | ---------------- |
| 480p       | 1000-1500 kbps      | Low bandwidth    |
| 720p       | 2500-4000 kbps      | Standard quality |
| 1080p      | 4500-6000 kbps      | High quality     |

### FFmpeg Presets

The script uses FFmpeg presets for encoding speed vs quality:

- `ultrafast`: Fastest encoding, larger file size
- `superfast`: Very fast encoding
- `veryfast`: Fast encoding (default)
- `faster`: Faster encoding
- `fast`: Balanced speed and quality

## Troubleshooting

### Authentication Issues

**Problem**: "Client secrets file not found"

- **Solution**: Ensure `client_secrets.json` is in the project root

**Problem**: "Failed to refresh credentials"

- **Solution**: Delete `youtube_token.pickle` and re-authenticate

### Streaming Issues

**Problem**: "FFmpeg not found"

- **Solution**: Install FFmpeg on your system (see prerequisites)

**Problem**: "Failed to create broadcast"

- **Solution**: Check your YouTube channel has live streaming enabled

**Problem**: "Stream quality is poor"

- **Solution**: Adjust `--youtube-bitrate` and `--youtube-resolution`

### YouTube API Limits

- **Daily quota**: 10,000 units per day (typical usage: ~100 units per stream)
- **Concurrent streams**: 1 concurrent stream for most channels
- **Stream duration**: Maximum 4 hours per stream (configurable)

## Stream Management

### During Streaming

1. The script will output the YouTube stream URL
2. Viewers can watch at: `https://www.youtube.com/watch?v=[BROADCAST_ID]`
3. Press `Ctrl+C` to stop streaming gracefully

### After Streaming

- The broadcast will automatically end
- The stream will be saved as a video on your channel (if recording is enabled)
- Stream statistics will be displayed

## Advanced Configuration

### Custom YouTube Config

You can modify the YouTube streaming configuration in `src/config/settings.py`:

```python
@dataclass
class YouTubeStreamingConfig:
    # Video encoding settings
    resolution: str = "720p"
    frame_rate: str = "30fps"
    video_bitrate: int = 2500

    # FFmpeg settings
    ffmpeg_preset: str = "veryfast"
    ffmpeg_crf: int = 23

    # Stream management
    max_stream_duration: int = 43200  # 12 hours
```

### Environment Variables

You can set default values using environment variables:

```bash
export YOUTUBE_CLIENT_SECRETS=/path/to/client_secrets.json
export YOUTUBE_DEFAULT_TITLE="My YOLO Stream"
export YOUTUBE_DEFAULT_BITRATE=3000
```

## Security Considerations

1. **Keep credentials secure**: Never commit `client_secrets.json` or `youtube_token.pickle`
2. **Limit API scope**: Only grant necessary permissions
3. **Use test accounts**: Test with a separate YouTube channel first
4. **Monitor usage**: Keep track of API quota usage

## Legal Considerations

1. **Content rights**: Ensure you have rights to stream the video content
2. **Privacy**: Be aware of privacy implications when streaming camera feeds
3. **YouTube policies**: Follow YouTube's community guidelines and terms of service
4. **Data protection**: Consider GDPR/privacy laws when streaming in public areas
