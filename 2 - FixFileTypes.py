import glob
import subprocess
import os
import re
import logging
import traceback
from random import randint
import imghdr
import PIL
from PIL import Image
import sys

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
        print(f"Found {extension} hiding as JPEG but couldn't rename:", file_obj)
        return

    print(f"Found {extension} hiding as JPEG, renaming:", file_obj, '->', new_file)

    subprocess.run(['mv', file_obj, new_file])


def get_frames_from_gif(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print
        "Cant load", infile
        sys.exit(1)

    i = 0

    try:
        while 1:
            im2 = im.convert('RGBA')
            im2.load()
            filename = os.path.join(os.path.dirname(infile), 'foo' + str(i) + '.jpg')
            background = Image.new("RGB", im2.size, (255, 255, 255))
            background.paste(im2, mask=im2.split()[3])
            background.save(filename, 'JPEG', quality=80)
            print(f"FOUND GIF, SAVING FRAME AS {filename}")
            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass  # end of sequence


for root, dirs, files in os.walk(directory):

    for file in files:

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

i = 1
for root, dirs, files in os.walk(directory):
    for file in files:
        try:
            file_obj = os.path.join(root, file)
            path, file_base_name = os.path.split(file_obj)
            old_path = os.path.splitext(file_base_name)
            old_ext = old_path[1]
            old_name = old_path[0]
            new_file = os.path.join(path, str(i) + "-" + str(random_with_N_digits(10)) + old_ext)
            if file_obj != new_file and "foo" not in old_name:
                print(f"Moving file\n"
                      f"{new_file}\n"
                      f"{file_obj}")
                subprocess.run(['mv', file_obj, new_file])
                i += 1
        except Exception as e:
            logging.error(traceback.format_exc())

print("Cleaning JPEGs done")
