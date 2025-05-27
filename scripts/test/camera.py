import cv2
import time
import einops
import numpy as np
from hsl_ur5.input.camera import OpenCVCamera, OpenCVCameraConfig

scene_cam_config = OpenCVCameraConfig(30, 640, 480, exposure=200)
scene_cam = OpenCVCamera(camera_index=2, config=scene_cam_config)
scene_cam.connect()

while True:
    # Capture each frame from the webcam
    frame = scene_cam.async_read()

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Display the frame in a window
    cv2.imshow("Webcam Feed", frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

# Release the webcam and close all OpenCV windows
cv2.destroyAllWindows()