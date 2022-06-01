import os
import logging
import traceback
import imghdr
import sys
import multiprocessing
import json
import shutil

from PIL import Image
from random import randint
from threading import Lock

directory = "downloads"


def random_with_N_digits(n):
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


def change_file_extension(file_obj, extension):
    old_path = os.path.splitext(file_obj)
    if not os.path.isfile(old_path[0] + extension):
        new_file = old_path[0] + extension
    elif not os.path.isfile(file_obj + extension):
        new_file = file_obj + extension
    else:
        return

    print(f"Found {extension} hiding as JPEG, renaming:", file_obj, '->', new_file)

    os.rename(file_obj, new_file)


def get_frames_from_gif(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)

    iterator = 0
    try:
        while 1:
            im2 = im.convert('RGBA')
            im2.load()
            filename = os.path.join(os.path.dirname(infile), 'foo' + str(i) + '.jpg')
            background = Image.new("RGB", im2.size, (255, 255, 255))
            background.paste(im2, mask=im2.split()[3])
            background.save(filename, 'JPEG', quality=80)
            iterator += 1
            while (iterator % 10 != 0):
                im.seek(im.tell() + 1)
    except EOFError:
        pass  # end of sequence


i = 1


def clean_image(file_root):
    root = file_root[0]
    file = file_root[1]
    try:
        file_obj = os.path.join(root, file)
        exten = os.path.splitext(file)[1].lower()
        img_type = imghdr.what(file_obj)
        # print(file_obj)
        if img_type is None:
            os.remove(file_obj)
        elif "jpeg" in img_type:
            if "jpeg" not in exten and "jpg" not in exten:
                change_file_extension(file_obj, ".jpeg")
        elif "png" in img_type:
            if "png" not in exten:
                change_file_extension(file_obj, ".png")
        elif "gif" in img_type:
            get_frames_from_gif(file_obj)
            os.remove(file_obj)
        else:
            os.remove(file_obj)
    except Exception as e:
        logging.error(traceback.format_exc())
    mutex.acquire()
    global i
    i += 1
    if i % 100 == 0:
        print("changing type" + str(i))
    mutex.release()


ii = 1


def rename_images(file_root):
    root = file_root[0]
    file = file_root[1]
    try:
        file_obj = os.path.join(root, file)
        path, file_base_name = os.path.split(file_obj)
        old_path = os.path.splitext(file_base_name)
        old_ext = old_path[1]
        old_name = old_path[0]
        mutex.acquire()
        global ii
        ii += 1
        new_file = os.path.join(path, str(ii) + "-" + str(random_with_N_digits(10)) + old_ext)
        if ii % 1000 == 0:
            print(f"Moving file"
                  f"{new_file}"
                  f"{file_obj} - {ii}")
        mutex.release()

        if file_obj != new_file and "foo" not in old_name:
            os.rename(file_obj, new_file)


    except Exception as e:
        logging.error(traceback.format_exc())


# recursively merge two folders including subfolders
def mergefolders(root_src_dir, root_dst_dir):
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)


def merge_pokemon(poke_dict):
    file_dir = {}
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            new = dir.replace(" pokemon", "")
            try:
                os.rename(os.path.join(root, dir), os.path.join(root, new))
            except Exception as e:
                print(e)
                os.remove(os.path.join(root, dir))
            file_dir[new] = os.path.join(root, dir)
    for pokemon, plist in poke_dict.items():
        poke = pokemon.replace("-", " ")
        print(poke)
        default_dir = file_dir[poke] if poke in file_dir else "./downloads/" + poke
        for item in plist:
            i = item.replace("-", " ")
            if i in file_dir and file_dir[i] != default_dir:

                print(f"merged {file_dir[i]} with {default_dir}")
                mergefolders(file_dir[i], default_dir)
                shutil.rmtree(file_dir[i])


mutex = Lock()

if __name__ == '__main__':

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    with open("pokemon-forms.json") as f:
        poke_dict = json.load(f)


    file_root_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_root_list.append((root, file))
    pool.map(clean_image, file_root_list)

    file_root_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_root_list.append((root, file))
    pool.map(rename_images, file_root_list)
    merge_pokemon(poke_dict)
    print("Cleaning JPEGs done")
