from tensorflow.contrib.keras.api import keras
from tensorflow.contrib import lite

keras_file = "weights.mobilenet.non-transfer.best.hdf5"
keras.models.load_model(keras_file)

h5_model = keras.models.load_model(keras_file)
converter = lite.TocoConverter.from_keras_model_file(keras_file)

tflite_model = converter.convert()
with open('mobilenet.tflite', 'wb') as f:
    f.write(tflite_model)
