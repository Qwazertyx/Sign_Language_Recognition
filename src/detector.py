import cv2
import numpy as np

# Try importing mediapipe, warn if not installed
try:
    import mediapipe as mp
except ImportError:
    mp = None


def detect_hand_hsv(frame_bgr):
    """
    Legacy: Detect hand region in a BGR frame using HSV color segmentation. Returns mask, bbox.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox = (x, y, w, h)
    return mask, bbox

class MediapipeHandDetector:
    """
    Hand detector using Mediapipe (Google). Provides bounding box and landmarks.
    """
    def __init__(self, max_num_hands=1, detection_confidence=0.7):
        if mp is None:
            raise ImportError("mediapipe is not installed! Install with 'pip install mediapipe'.")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         min_detection_confidence=detection_confidence)

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            # Get bounding box covering all landmarks for the first detected hand
            h, w, _ = frame_bgr.shape
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark], dtype=np.int32)
            x, y, w_box, h_box = cv2.boundingRect(landmarks)
            return (x, y, w_box, h_box), landmarks  # bbox, landmarks
        else:
            return None, None
