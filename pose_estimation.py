import cv2 as cv
import mediapipe as mp
import numpy as np
import time

cap = cv.VideoCapture('2.mp4')
if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

# Mediapipe pose setup
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(
    static_image_mode=False,        
    model_complexity=2,             
    smooth_landmarks=True,          
    min_detection_confidence=0.9,   
    min_tracking_confidence=0.9    
)

# To calculate FPS
prev_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("End of video or cannot read the frame.")
        break

    # Preprocess: Resize and enhance frame
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    img = cv.resize(img, (new_width, new_height))
    img = cv.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Pose detection
    results = pose.process(img_rgb)

    # Draw pose landmarks and connections
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if lm.visibility > 0.5:  # Only draw visible keypoints
                cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
                
                
    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv.putText(img, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the output
    cv.imshow('Pose Detection', img)

    # Quit if 'q' or 'Q' is pressed
    if cv.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

# Release resources
cap.release()
cv.destroyAllWindows()
