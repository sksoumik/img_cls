from data_generator import create_data_generator, count_images
from model import build_model

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
