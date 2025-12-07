import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_model_big.h5')
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'nothing', 'space'  # as in test dataset
]
IMG_SIZE = (64, 64)

class Predictor:
    def __init__(self, model_path=MODEL_PATH, class_names=CLASS_NAMES):
        self.model = load_model(model_path)
        self.class_names = class_names

    def preprocess(self, frame, bbox):
        x, y, w, h = bbox
        hand_img = frame[y:y+h, x:x+w]
        try:
            img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)).resize(IMG_SIZE)
        except Exception:
            img = Image.fromarray(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8))
        arr = np.array(img) / 255.0
        return np.expand_dims(arr, axis=0)  # batch dimension

    def predict_from_frame(self, frame, bbox):
        inp = self.preprocess(frame, bbox)
        preds = self.model.predict(inp, verbose=0)
        idx = np.argmax(preds)
        return self.class_names[idx], float(preds[0][idx])
