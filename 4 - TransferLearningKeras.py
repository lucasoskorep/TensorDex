from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import ImageFile
from tensorflow import keras

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_shape = (224, 224, 3)

batch_size = 64
model_name = "TF2_Mobilenet_V2_transfer"

# preproc = keras.applications.inception_v3.preprocess_input
preproc = keras.applications.mobilenet_v2.preprocess_input

train_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rescale=1. / 255,
    # rotation_range=30,
    # width_shift_range=[-.1, .1],
    # height_shift_range=[-.1, .1],
    # preprocessing_function=preproc
)
train_gen = train_idg.flow_from_directory(
    './downloads',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode='rgb'
)

val_idg = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rescale=1. / 255,
    # rotation_range=30,
    # width_shift_range=[-.1, .1],
    # height_shift_range=[-.1, .1],
    # preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
)
val_gen = val_idg.flow_from_directory(
    './data/val',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)

print((val_gen.classes))
nclass = len(train_gen.class_indices)
print(nclass)
# for _ in range(5):
#     img, label = train_gen.next()
#     print(img.shape)  # (1,256,256,3)
#     plt.imshow(img[0])
#     plt.show()
# plt.imshow(

# base_model = vgg16.VGG16(
#     weights='imagenet',
#     include_top=False,
#     input_shape=input_shape
# )
# base_model = keras.applications.InceptionV3(
#     weights='imagenet',
#     include_top=False,
#     input_shape=input_shape
# )
# base_model = keras.applications.xception.Xception(
#     weights='imagenet',
#     include_top=False,
#     input_shape=input_shape
# )

base_model = keras.applications.mobilenet_v2.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)
base_model.trainable = False
# i = keras.layers.Input([input_shape[0], input_shape[1], input_shape[2]])
i = base_model.input
# x = preproc(i)
# x = base_model
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
output = keras.layers.Dense(nclass, activation='softmax')(x)

model = keras.Model(inputs=i, outputs=output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0001),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
print(model.output_shape)

# Train the model
file_path = "weights.mobilenet.best.hdf5"

checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True,
                                             mode='min')

early = keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=15)

tensorboard = keras.callbacks.TensorBoard(
    log_dir="logs/" + model_name + "{}".format(time()),
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq=1,
    profile_batch=2,
    embeddings_freq=1
)

callbacks_list = [checkpoint, early, tensorboard]  # early

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    batch_size=batch_size,
    shuffle=True,
    verbose=True,
    callbacks=callbacks_list
)

# Create Test generator
test_idg = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)

test_gen = test_idg.flow_from_directory(
    './data/test',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False
)

len(test_gen.filenames)

score = model.evaluate_generator(test_gen, workers=1, steps=len(test_gen))

# predicts
predicts = model.predict_generator(test_gen, verbose=True, workers=1, steps=len(test_gen))

print("Loss: ", score[0], "Accuracy: ", score[1])
print(score)

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
