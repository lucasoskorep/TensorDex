import os
from random import random
from shutil import copyfile, rmtree

train_dir = "./data/train/"
test_dir = "./data/test/"
val_dir = "./data/val/"
train = .75
test = .20
val = .05


def add_train_data(file, filename, label):
    dest = train_dir + label + "/" + filename
    print(dest, label, filename)
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)
    try:
        copyfile(file, dest)
    except Exception as e:
        print(e)
        print("INVALID FILE")
        os.remove(file)
        # TODO: Remove the files


def add_val_data(file, filename, label):
    dest = val_dir + label + "/" + filename
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)
    copyfile(file, dest)


def add_test_data(file, filename, label):
    dest = test_dir + label + "/" + filename
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)
    copyfile(file, dest)


def remove_previous():
    if os.path.exists(os.path.dirname(test_dir)):
        rmtree(test_dir)
    if os.path.exists(os.path.dirname(train_dir)):
        rmtree(train_dir)
    if os.path.exists(os.path.dirname(val_dir)):
        rmtree(val_dir)


remove_previous()
files_processed = 0

for root, dirs, files in os.walk("downloads/"):

    for file in files:
        print(file)

        if file is ".DS_Store":
            continue
        c = random()

        if c < train:
            add_train_data(os.path.join(root, file), file, root.split("/")[-1])
        elif c < (train + val):
            add_val_data(os.path.join(root, file), file, root.split("/")[-1])
        else:
            add_test_data(os.path.join(root, file), file, root.split("/")[-1])
        files_processed += 1
        print(root.split("/")[-1])
        print(files_processed)
        print(file)
