import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from glob import glob

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


accuracies = []
losses = []
filenames = []

input_shape = (224, 224, 3)
batch_size = 32

test_gen = ImageDataGenerator().flow_from_directory(
    './data/test',
    # './single_image_test_set',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False
)

for file  in glob("./models/keras/*"):
    filenames.append(file)
    print(file)
    model = load_model(file)

    predictions = model.predict(test_gen, verbose=True, workers=12)

    print(predictions)
    print(type(predictions))
    print(predictions.shape)

    # Process the predictions
    predictions = np.argmax(predictions,
                            axis=1)
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

    accuracies.append(acc)

overall_df = pd.DataFrame(list(zip(filenames, accuracies)),
               columns =['model', 'acc']).sort_values('acc')

print(overall_df)
overall_df.plot.bar(y="acc", rot=90)
plt.tight_layout()
plt.show()
overall_df.to_csv("all_model_output.csv")
