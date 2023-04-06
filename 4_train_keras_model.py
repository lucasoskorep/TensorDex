from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow import keras

from modeling_utils import ImageClassModelBuilder, ImageClassModels

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_shape = (224, 224, 3)

batch_size = 32

training_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=[-.1, .1],
    height_shift_range=[-.1, .1],
)
val_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
)
testing_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
)


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def get_gen(path, dataset_type: DatasetType = DatasetType.TRAIN):
    idg = None
    if dataset_type is DatasetType.TRAIN:
        idg = training_idg
    if dataset_type is DatasetType.TEST:
        idg = testing_idg
    if dataset_type is DatasetType.VAL:
        idg = val_idg

    return idg.flow_from_directory(
        path,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )


def train_model(model, model_name, train_gen, val_gen, max_epochs):
    print(model)
    print(f"NOW TRAINING: {model_name}")
    checkpoint = keras.callbacks.ModelCheckpoint(
        f"./models/keras/{model_name}.hdf5",
        monitor='val_categorical_crossentropy',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_categorical_crossentropy",
        mode="auto",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    tensorboard = keras.callbacks.TensorBoard(
        log_dir="logs/" + model_name,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq=1,
        profile_batch=2,
        embeddings_freq=1,
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=True,
        workers=20,
        callbacks=[checkpoint, early, tensorboard],
        max_queue_size=1000
    )
    return model


def test_model(model, test_gen):
    predictions = model.predict(test_gen, verbose=True, workers=1, steps=len(test_gen))

    print(predictions)
    print(type(predictions))
    print(predictions.shape)
    # Process the predictions
    predictions = np.argmax(predictions,
                            axis=1)
    # test_gen.reset()
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    predictions = [label_index[p] for p in predictions]
    reals = [label_index[p] for p in test_gen.classes]

    # Processed the saved results
    acc = accuracy_score(reals, predictions)
    conf_mat = confusion_matrix(reals, predictions)
    print(classification_report(reals, predictions, labels=[l for l in label_index.values()]))
    print("Testing accuracy score is ", acc)
    print("Confusion Matrix", conf_mat)

    print("made dataframe")
    plt.figure(figsize=(10, 7))
    print("made plot")
    print("showing plot")
    plt.show()


if __name__ == "__main__":
    model_builders = [
        ImageClassModelBuilder(
            input_shape=input_shape,
            n_classes=807,
            optimizer=keras.optimizers.Adam(learning_rate=.0001),
            pre_trained=True,
            freeze_layers=True,
            freeze_batch_norm=True,
            base_model_type=ImageClassModels.MOBILENET_V2,
            dense_layer_neurons=1024,
            dropout_rate=.5,
        ),        ImageClassModelBuilder(
            input_shape=input_shape,
            n_classes=807,
            optimizer=keras.optimizers.Adam(learning_rate=.0001),
            pre_trained=True,
            freeze_layers=True,
            freeze_batch_norm=True,
            base_model_type=ImageClassModels.INCEPTION_RESNET_V2,
            dense_layer_neurons=1024,
            dropout_rate=.5,
        ),        ImageClassModelBuilder(
            input_shape=input_shape,
            n_classes=807,
            optimizer=keras.optimizers.Adam(learning_rate=.0001),
            pre_trained=True,
            freeze_layers=True,
            freeze_batch_norm=True,
            base_model_type=ImageClassModels.INCEPTION_V3,
            dense_layer_neurons=1024,
            dropout_rate=.5,
        ),        ImageClassModelBuilder(
            input_shape=input_shape,
            n_classes=807,
            optimizer=keras.optimizers.Adam(learning_rate=.0001),
            pre_trained=True,
            freeze_layers=True,
            freeze_batch_norm=True,
            base_model_type=ImageClassModels.XCEPTION,
            dense_layer_neurons=1024,
            dropout_rate=.5,
        ),        ImageClassModelBuilder(
            input_shape=input_shape,
            n_classes=807,
            optimizer=keras.optimizers.Adam(learning_rate=.0001),
            pre_trained=True,
            freeze_layers=True,
            freeze_batch_norm=True,
            base_model_type=ImageClassModels.DENSENET201,
            dense_layer_neurons=1024,
            dropout_rate=.5,
        )
    ]
    for mb in model_builders:
        model = mb.create_model()
        model_name = mb.get_name()
        train_gen = get_gen('./data/train', dataset_type=DatasetType.TRAIN)
        val_gen = get_gen('./data/val', dataset_type=DatasetType.VAL)
        test_gen = get_gen('./data/test', dataset_type=DatasetType.TEST)
        model = train_model(model, model_name, train_gen, val_gen, 1)
        # for layer in model.layers[2].layers:
        #     if not isinstance(layer, keras.layers.BatchNormalization):
        #         layer.trainable = True
        # model.layers[2].trainable = True
        # print(model)
        # model.compile(
        #     optimizer=keras.optimizers.Adam(learning_rate=.00001),
        #     loss=keras.losses.CategoricalCrossentropy(),
        #     metrics=['accuracy', 'categorical_crossentropy']
        # )
        # model.summary()
        # model = train_model(model, model_name + "-second_stage", train_gen, val_gen, 1)
        # test_model(model, test_gen)
