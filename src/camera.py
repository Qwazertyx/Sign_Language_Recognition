import cv2
from . import detector
from .predictor import Predictor

# Initialize detector
mp_detector = None
mp_error = None
try:
    mp_detector = detector.MediapipeHandDetector()
except Exception as e:
    mp_error = str(e)

# Initialize model predictor
predictor = Predictor()

def run_camera_demo():
    """
    Opens the webcam, shows live video in a window with hand detection (Mediapipe or HSV) and real-time prediction.
    Press 'q' to exit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    print("Press 'q' to exit webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        display_frame = frame.copy()
        hand_detected = False
        pred_str = ''
        prob = 0.0
        pred_pos = (30, 70)

        if mp_detector:
            bbox, landmarks = mp_detector.detect(frame)
            if bbox:
                x, y, w, h = bbox
                hand_detected = True
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, 'Hand (Mediapipe)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
                # Draw landmarks
                for lx, ly in landmarks:
                    cv2.circle(display_frame, (lx, ly), 3, (255,0,0), -1)
                # Predict
                pred_str, prob = predictor.predict_from_frame(frame, bbox)
                pred_pos = (x, y+h+30)
        else:
            if mp_error:
                cv2.putText(display_frame, f'Mediapipe error: {mp_error}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Fallback: HSV
            hand_mask, bbox = detector.detect_hand_hsv(frame)
            if bbox:
                x, y, w, h = bbox
                hand_detected = True
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0,165,255), 2)
                cv2.putText(display_frame, 'Hand (HSV)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)
                pred_str, prob = predictor.predict_from_frame(frame, bbox)
                pred_pos = (x, y+h+30)
            cv2.imshow('Hand Mask (HSV)', hand_mask)
        
        # If a prediction
        if hand_detected and pred_str:
            cv2.putText(
                display_frame,
                f'Predicted: {pred_str} ({prob:.2f})',
                pred_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                3
            )
        elif not hand_detected and not mp_error:
            cv2.putText(display_frame, 'No Hand Detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow('Webcam - Sign Recognition', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
