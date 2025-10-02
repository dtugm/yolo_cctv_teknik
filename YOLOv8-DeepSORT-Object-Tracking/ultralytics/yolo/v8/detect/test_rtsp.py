import cv2

# Test RTSP connection
rtsp_url = "rtsp://10.2.10.70:7447/owNskP1rv1LKe2mV"
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not open RTSP stream")
else:
    print("RTSP stream opened successfully")
    
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i}: {frame.shape}")
            cv2.imshow('RTSP Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Failed to read frame {i}")
    
    cap.release()
    cv2.destroyAllWindows()
