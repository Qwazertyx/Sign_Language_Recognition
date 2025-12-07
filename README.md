# Sign Language Recognition (Real-time, Webcam)

Recognize American Sign Language hand poses (A-Z) live using your webcam and a deep neural network.

## How it works
- Detects your hand in webcam video (Mediapipe/HSV).
- Crops and feeds to a CNN model trained on a large hand sign dataset.
- Shows the predicted letter and confidence live.

## Setup
1. **Requirements**:
```bash
pip install opencv-python mediapipe numpy pillow tensorflow
```
2. **Dataset**:
   - Place your dataset folder (subfolders per class/letter, e.g. `a/`, `b/`, ...) inside:
     ```
     data/asl_big_dataset/asl_dataset/
     ```
   - You can download a dataset (e.g. ASL Alphabet) from Kaggle or other open resources.

## Training
Generate a sign recognition model from your dataset (run this once):
```bash
python -m src.model
```
This will output `models/sign_model_big.h5`.

## Running Live Recognition
With model and webcam ready, start:
```bash
python main.py
```
Press 'q' to quit the webcam window.

## Example Demo Output

![Hand sign demo](Screenshot%20K.png)

_Above: model detects hand, landmarks, and classifies a "K" sign from webcam._
