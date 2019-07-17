import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from keras import optimizers
from keras.applications import inception_v3, mobilenet_v2, vgg16
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from time import time
from PIL import ImageFile

# First we some globals that we want to use for this entire process

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_shape = (224, 224, 3)
batch_size = 32

model_name = "mobilenet-fixed-data"

# Next we set up the Image Data Generators to feed into the training cycles.
# We need one for training, validation, and testing
train_idg = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=[-.1, .1],
    height_shift_range=[-.1, .1],
    preprocessing_function=preprocess_input
)

train_gen = train_idg.flow_from_directory(
    './data/train',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size
)

print(len(train_gen.classes))

val_idg = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=[-.1, .1],
    height_shift_range=[-.1, .1],
    preprocessing_function=preprocess_input
)

val_gen = val_idg.flow_from_directory(
    './data/test',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size
)

test_idg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)
test_gen = test_idg.flow_from_directory(
    './data/test',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False

)

# Now we define the model we are going to use....to use something differnet just comment it out or add it here

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
    # weights='imagenet',
    include_top=False,
    input_shape=input_shape
)


# Create a new top for that model
add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
# add_model.add(Dense(4048, activation='relu'))
# add_model.add(Dropout(0.5))

add_model.add(Dense(2024, activation='relu'))
# Adding some dense layers in order to learn complex functions from the base model
# Potentially throw another dropout layer here if you seem to be overfitting your
add_model.add(Dropout(0.5))
add_model.add(Dense(512, activation='relu'))
add_model.add(Dense(len(train_gen.class_indices), activation='softmax'))  # Decision layer

#TODO: Add in gpu support
model = multi_gpu_model(add_model, 2)
# model = add_model

model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])
model.summary()




# Now that the model is created we can go ahead and train on it using the image generators we created earlier

file_path = model_name + ".hdf5"

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
    epochs=25,
    shuffle=True,
    verbose=True,
    callbacks=callbacks_list
)




# Finally we are going to grab predictions from our model, save it, and then run some analysis on the results

predicts = model.predict_generator(test_gen, verbose=True, workers=1, steps=len(test_gen))

keras_file = model_name + 'finished.h5'
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

acc = accuracy_score(reals, predicts)
conf_mat = confusion_matrix(reals, predicts)
print(classification_report(reals, predicts, [l for l in label_index.values()]))
print("Testing accuracy score is ", acc)
print("Confusion Matrix", conf_mat)

df_cm = pd.DataFrame(conf_mat, index=[i for i in list(set(reals))],
                     columns=[i for i in list(set(reals))])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()

with open("labels.txt", "w") as f:
    for label in label_index.values():
        f.write(label + "\n")
