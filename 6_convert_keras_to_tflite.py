import os
from glob import glob
from pathlib import Path

import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# TODO: Move these to a config for the project
input_shape = (224, 224, 3)
batch_size = 32

single_gen = ImageDataGenerator().flow_from_directory(
    './single_image_test_set',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False
)

pd.DataFrame(sorted([f.name for f in os.scandir("./data/train") if f.is_dir()])).to_csv("./models/tflite/labels.txt",
                                                                                        index=False, header=False)

for file in glob("./models/keras/*.hdf5"):
    path = Path(file)
    tflite_file = f'./models/tflite/models/{path.name[:-5] + ".tflite"}'
    if not Path(tflite_file).exists():
        print(tflite_file)
        keras_model = tf.keras.models.load_model(file)
        keras_model.summary()
        print(keras_model.input)
        print(keras_model.layers)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        with open(tflite_file, 'wb') as f:
            f.write(tflite_model)
    # TODO: Verify the model performance after converting to TFLITE
    # interpreter = tf.lite.Interpreter(model_path=tflite_file)
    # single_acc, single_ll = get_metrics(single_gen, keras_model)
    # tf_single_acc, tf_single_ll = get_metrics(single_gen, tflite_model)
    #
    # print(single_acc, tf_single_acc)
    # print(single_ll, tf_single_ll)
