from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Lambda,
    Input,
    GlobalAveragePooling2D,
)


def build_model(input_shape, num_classes):
    before_mobilenet = Sequential(
        [
            Input(shape=input_shape),
            Lambda(preprocess_input),
        ]
    )

    mobilenet = MobileNetV2(input_shape=input_shape, include_top=False)

    after_mobilenet = Sequential(
        [
            GlobalAveragePooling2D(),
            Dropout(0.1),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model = Sequential([before_mobilenet, mobilenet, after_mobilenet])

    return model
