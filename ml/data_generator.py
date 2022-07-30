from random import shuffle
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def count_images(TARGET_DIR):
    for class_name in os.listdir(TARGET_DIR):
        class_path = os.path.join(TARGET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        print("{}: {}".format(class_name, len(os.listdir(class_path))))


def create_data_generator(train_data_path, valid_data_path, test_data_path):
    train_data_gen = ImageDataGenerator(
        horizontal_flip=True,
    )
    train_generator = train_data_gen.flow_from_directory(
        directory=train_data_path,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

    val_data_gen = ImageDataGenerator()
    validation_generator = val_data_gen.flow_from_directory(
        directory=valid_data_path,
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator
