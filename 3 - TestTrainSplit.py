import os
from random import random
from shutil import copyfile, rmtree
from pathlib import Path
import multiprocessing

train_dir = "./data/train/"
test_dir = "./data/test/"
val_dir = "./data/val/"
train = .80
test = .10
val = .10


def add_train_data(file, filename, label):
    dest = train_dir + label + "/" + filename
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)
    try:
        Path(dest).absolute().symlink_to(Path(file).absolute())
    except Exception as e:
        print(e)
        print("INVALID FILE")
        os.remove(file)


def add_val_data(file, filename, label):
    dest = val_dir + label + "/" + filename
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)

    Path(dest).absolute().symlink_to(Path(file).absolute())


def add_test_data(file, filename, label):
    dest = test_dir + label + "/" + filename
    if not os.path.exists(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except Exception as e:
            print(e)

    Path(dest).absolute().symlink_to(Path(file).absolute())


def remove_previous():
    if os.path.exists(os.path.dirname(test_dir)):
        rmtree(test_dir)
    if os.path.exists(os.path.dirname(train_dir)):
        rmtree(train_dir)
    if os.path.exists(os.path.dirname(val_dir)):
        rmtree(val_dir)


files_processed = 0
def test_split_file(file_root):
    global files_processed
    root = file_root[0]
    file = file_root[1]
    # print(file)

    if file == ".DS_Store":
        return
    c = random()

    if c < train:
        add_train_data(os.path.join(root, file), file, root.split("/")[-1])
    elif c < (train + val):
        add_val_data(os.path.join(root, file), file, root.split("/")[-1])
    else:
        add_test_data(os.path.join(root, file), file, root.split("/")[-1])
    files_processed += 1

    if files_processed % 1000==0:
        print(root.split("/")[-1])
        print(files_processed)
        print(file)


if __name__ == '__main__':
    remove_previous()

    file_root_list = []

    for root, dirs, files in os.walk("downloads/"):
        for file in files:
            file_root_list.append((root, file))


    pool = multiprocessing.Pool(multiprocessing.cpu_count()*2)

    pool.map(test_split_file, file_root_list)

