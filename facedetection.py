import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize VideoCapture object
cap = cv2.VideoCapture(0)

# Initialize face detection model
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Flip the frame horizontally for a more natural viewing experience
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use Mediapipe to detect faces in the frame
        results = face_detection.process(frame_rgb)

        # Draw the face detection annotations on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        else:
            pass

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
