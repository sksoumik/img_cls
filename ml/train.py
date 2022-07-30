from numpy import save
from data_generator import create_data_generator, count_images
from model import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf


def train_model(model, train_generator, validation_generator):
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[
            ModelCheckpoint(
                "../model/",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                verbose=1,
                mode="min",
            ),
        ],
    )

    return model


if __name__ == "__main__":
    TRAIN_DIR = "../dataset/seg_train/seg_train/"
    VALIDATION_DIR = "../dataset/seg_test/seg_test/"
    TEST_DIR = "../dataset/seg_pred/seg_pred/"

    train_generator, validation_generator = create_data_generator(
        train_data_path=TRAIN_DIR,
        valid_data_path=VALIDATION_DIR,
        test_data_path=TEST_DIR,
    )

    print("\nTrain data image counts class-wise:")
    count_images(TRAIN_DIR)

    print("\nValidation data image counts class-wise:")
    count_images(VALIDATION_DIR)

    labels = train_generator.class_indices
    class_mapping = dict((v, k) for k, v in labels.items())
    print(f"\nClass mapping: {class_mapping}")

    input_shape = (150, 150, 3)
    num_classes = len(class_mapping)
    model = build_model(input_shape, num_classes)
    model = train_model(model, train_generator, validation_generator)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("../model/model.tflite", "wb").write(tflite_model)
