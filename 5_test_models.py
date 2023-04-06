from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageFile
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from modeling_utils import get_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

accuracies = []
losses = []
filenames = []

input_shape = (224, 224, 3)
batch_size = 32
metrics_df = pd.read_csv("all_model_output.csv")

test_gen = ImageDataGenerator().flow_from_directory(
    './data/test',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False
)

single_gen = ImageDataGenerator().flow_from_directory(
    './single_image_test_set',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    shuffle=False
)

for file in glob("./models/keras/*.hdf5"):
    print(file)
    if file in metrics_df.values:
        continue
    model = load_model(file)
    test_acc, test_ll = get_metrics(test_gen, model)
    single_acc, single_ll = get_metrics(single_gen, model, file[:-5] + ".csv")
    metrics_df = metrics_df.append({
        "model": file,
        "test_acc": test_acc,
        "test_loss": test_ll,
        "single_acc": single_acc,
        "single_loss": single_ll,
    }, ignore_index=True)



# Save the results

metrics_df.to_csv("all_model_output.csv", index=False)
print(metrics_df)
metrics_df = metrics_df.sort_values('single_acc')
metrics_df.plot.bar(y=["test_acc", "single_acc"], rot=90)
metrics_df = metrics_df.sort_values('test_acc')
metrics_df.plot.bar(y=["test_acc", "single_acc"], rot=90)
plt.tight_layout()
plt.show()
