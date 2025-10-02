import cv2

def test_rtsp_connection():
    rtsp_url = "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV"
    
    print(f"Testing connection to: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Failed to connect to RTSP stream")
        return False
    
    print("Connection successful!")
    
    frame_count = 0
    while frame_count < 100:  # Test for 100 frames
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            cv2.imshow('RTSP Test', frame)
            print(f"Frame {frame_count} received")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to read frame")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

# Run the test
test_rtsp_connection()
