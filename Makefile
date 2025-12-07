.PHONY: install train run clean

install:
	pip install opencv-python mediapipe numpy pillow tensorflow

train:
	python -m src.model

run:
	python main.py

clean:
	del models\sign_model_big.h5 2>NUL || true
