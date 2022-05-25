import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

model = load_model("./Models/mobilenetv2-stock-all-fixed-v2/mobilenetv2.hdf5")

input_shape = (224, 224, 3)
batch_size = 96

test_idg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

test_gen = test_idg.flow_from_directory(
    # './data/test',
    './SingleImageTestSet',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False

)

predictions = model.predict_generator(test_gen, verbose=True, workers=1, steps=len(test_gen))

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

# Save the results
print(label_index)
print(test_gen.classes)
print(test_gen.classes.shape)
print(type(test_gen.classes))
df = pd.DataFrame(columns=['fname', 'prediction', 'true_val'])
df['fname'] = [x for x in test_gen.filenames]
df['prediction'] = predictions
df["true_val"] = reals
df.to_csv("sub1_non_transfer.csv", index=False)

# Processed the saved results

acc = accuracy_score(reals, predictions)
conf_mat = confusion_matrix(reals, predictions)
print(classification_report(reals, predictions, labels=[l for l in label_index.values()]))
print("Testing accuracy score is ", acc)
print("Confusion Matrix", conf_mat)

df_cm = pd.DataFrame(conf_mat, index=[i for i in list(set(reals))],
                     columns=[i for i in list(set(reals))])
print("made dataframe")
plt.figure(figsize=(10, 7))
print("made plot")
# sn.heatmap(df_cm, annot=True)
print("showing plot")
plt.show()

with open("labels.txt", "w") as f:
    for label in label_index.values():
        f.write(label + "\n")

