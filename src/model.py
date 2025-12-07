import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'asl_big_dataset', 'asl_dataset')
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 14
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_model_big.h5')


def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    num_classes = train_gen.num_classes
    model = keras.Sequential([
        layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    model.save(MODEL_OUT)
    print(f'Model trained and saved to {MODEL_OUT}')
    print('Class labels:', list(train_gen.class_indices.keys()))

if __name__ == "__main__":
    train_model()
