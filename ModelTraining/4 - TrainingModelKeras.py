import tensorflow as tf
import pandas as pd
import numpy as np
import os
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow import keras
from time import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

input_shape = (299, 299, 3)
batch_size = 32
model_name = "MobileNetV2FullDatasetNoTransfer"

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

train_idg = ImageDataGenerator(
    # horizontal_flip=True,
    preprocessing_function=preprocess_input
)
train_gen = train_idg.flow_from_directory(
    './data/train',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size
)

val_idg = ImageDataGenerator(
    # horizontal_flip=True,
    preprocessing_function=preprocess_input
)

val_gen = val_idg.flow_from_directory(
    './data/val',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size
)
from keras.applications import inception_v3, mobilenet_v2, vgg16
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import optimizers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

nclass = len(train_gen.class_indices)

# base_model = vgg16.VGG16(
#     weights='imagenet',
#     include_top=False,
#     input_shape=input_shape
# )
# base_model = inception_v3.InceptionV3(
#     weights='imagenet',
#     include_top=False,
#     input_shape=input_shape
# )

base_model = mobilenet_v2.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(
    Dense(1024, activation='relu'))  # Adding some dense layers in order to learn complex functions from the base model
# Potentially throw another dropout layer here if you seem to be overfitting your
add_model.add(Dropout(0.5))
add_model.add(Dense(512, activation='relu'))
add_model.add(Dense(nclass, activation='softmax'))  # Decision layer

model = add_model
model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])
model.summary()

# Train the model
file_path = "weights.mobilenet.non-transfer.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_acc", mode="max", patience=15)

tensorboard = TensorBoard(
    log_dir="logs/" + model_name + "{}".format(time()), histogram_freq=0, batch_size=batch_size,
    write_graph=True,
    write_grads=True,
    write_images=True,
    update_freq=batch_size
)

callbacks_list = [checkpoint, early, tensorboard]  # early

history = model.fit_generator(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    epochs=2,
    shuffle=True,
    verbose=True,
    callbacks=callbacks_list
)

# Create Test generator
test_idg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)
test_gen = test_idg.flow_from_directory(
    './data/test',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False

)
len(test_gen.filenames)


# predicts
predicts = model.predict_generator(test_gen, verbose=True, workers=1, steps=len(test_gen))


keras_file = 'finished.h5'
keras.models.save_model(model, keras_file)

print(predicts)
print(type(predicts))
print(predicts.shape)
# Process the predictions
predicts = np.argmax(predicts,
                     axis=1)
# test_gen.reset()
label_index = {v: k for k, v in train_gen.class_indices.items()}
predicts = [label_index[p] for p in predicts]
reals = [label_index[p] for p in test_gen.classes]

# Save the results
print(label_index)
print(test_gen.classes)
print(test_gen.classes.shape)
print(type(test_gen.classes))
df = pd.DataFrame(columns=['fname', 'prediction', 'true_val'])
df['fname'] = [x for x in test_gen.filenames]
df['prediction'] = predicts
df["true_val"] = reals
df.to_csv("sub1_non_transfer.csv", index=False)

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

