import tensorflow as tf
import pandas as pd
import numpy as np
import os
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix


def print_preds(reals, preds):
    acc = accuracy_score(reals, predicts)
    conf_mat = confusion_matrix(reals, predicts)
    print("Testing accuracy score is ", acc)
    print("Confusion Matrix", conf_mat)

    df_cm = pd.DataFrame(conf_mat, index=[i for i in ["Block", "Meter", "Sign"]],
                         columns=[i for i in ["Block", "Meter", "Sign"]])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


data = pd.read_csv("sub1_non_transfer.csv")
files_list = list(data["fname"])
reals = list(data["true_val"])
predicts = list(data["prediction"])

reals2 = []
wrong_files = []
for root, dirs, files in os.walk(".\\photos"):
    for file in files:
        if file in files_list:
            x = data.loc[data["fname"] == file].values[0]
            if (x[1] != x[2]):
                print(x)
                wrong_files.append((os.path.join(root, file), x[1]))
            reals2.append(root.split("\\")[-1])

print_preds(reals, predicts)
print_preds(reals2, predicts)

import matplotlib.image as mpimg
from shutil import copyfile, rmtree

for file, pred in wrong_files:
    print(file)
    # img = mpimg.imread(file)
    # # end
    # # from now on you can use img as an image, but make sure you know what you are doing!
    # imgplot = plt.imshow(img)
    dest = file.split("\\")
    dest[1] = "failed"
    dest[-1] = pred + dest[-1]
    dest = "\\".join(dest)
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)
    copyfile(file, dest)
    plt.show()
