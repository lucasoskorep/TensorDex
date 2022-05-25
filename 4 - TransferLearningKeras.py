from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import ImageFile
from tensorflow import keras

from model_builders import ImageClassModelBuilder, ImageClassModels

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_shape = (224, 224, 3)

batch_size = 32
model_name = f"mobilenetv2-dense1024-l1l2-25drop-{time()}"

training_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=[-.1, .1],
    height_shift_range=[-.1, .1],
)
testing_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
)


def get_gen(path, test_set=False):
    idg = testing_idg if test_set else training_idg
    return idg.flow_from_directory(
        path,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )


def train_model(train_gen, val_gen):
    model = ImageClassModelBuilder(
        input_shape=input_shape,
        n_classes=807,
        optimizer=keras.optimizers.Adam(learning_rate=.0001),
        pre_trained=True,
        fine_tune=0,
        base_model=ImageClassModels.MOBILENET_V2
    ).create_model()
    # Train the model
    checkpoint = keras.callbacks.ModelCheckpoint(f"./Models/keras/{model_name}.hdf5", monitor='val_loss', verbose=1,
                                                 save_best_only=True,
                                                 mode='min')
    early = keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=15)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir="logs/" + model_name,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq=1,
        profile_batch=2,
        embeddings_freq=1,
    )
    callbacks_list = [checkpoint, early, tensorboard]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        batch_size=batch_size,
        shuffle=True,
        verbose=True,
        workers=12,
        callbacks=callbacks_list,
        max_queue_size=1000
    )
    print(history)
    return model


def test_model(model, test_gen):
    print(len(test_gen.filenames))
    score = model.evaluate(test_gen, workers=8, steps=len(test_gen))
    predicts = model.predict(test_gen, verbose=True, workers=8, steps=len(test_gen))
    print("Loss: ", score[0], "Accuracy: ", score[1])
    print(score)
    print(predicts)
    print(type(predicts))
    print(predicts.shape)

    # Process the predictions
    predicts = np.argmax(predicts,
                         axis=1)
    label_index = {v: k for k, v in test_gen.class_indices.items()}
    predicts = [label_index[p] for p in predicts]
    reals = [label_index[p] for p in test_gen.classes]

    # Save the results
    df = pd.DataFrame(columns=['fname', 'prediction', 'true_val'])
    df['fname'] = [x for x in test_gen.filenames]
    df['prediction'] = predicts
    df["true_val"] = reals
    df.to_csv("sub1.csv", index=False)
    # Processed the saved results
    from sklearn.metrics import accuracy_score, confusion_matrix
    acc = accuracy_score(reals, predicts)
    conf_mat = confusion_matrix(reals, predicts)
    print("Testing accuracy score is ", acc)
    print("Confusion Matrix", conf_mat)
    df_cm = pd.DataFrame(conf_mat, index=[i for i in list(set(reals))],
                         columns=[i for i in list(set(reals))])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


if __name__ == "__main__":
    train_gen = get_gen('./data/train')
    val_gen = get_gen('./data/val')
    test_gen = get_gen('./data/test', test_set=True)
    model = train_model(train_gen, val_gen)
    test_model(model, test_gen)
