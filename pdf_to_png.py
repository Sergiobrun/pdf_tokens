from pdf2image import convert_from_path
import glob, os
import time

directory = 'C:/Users/Usuario/PycharmProjects/biedronka/pdfs/'
for file in glob.glob(os.path.join(directory, "*.pdf")):
    print(file)
    images = convert_from_path(file)
    for i, image in enumerate(images):
        print(image)
        image.save(f'{file}_{i}.png')


import glob
import shutil
import os

src_dir = "C:/Users/Usuario/PycharmProjects/biedronka/pdfs/"
dst_dir = "C:/Users/Usuario/PycharmProjects/biedronka/pngs/"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.png")):
    shutil.move(jpgfile, dst_dir)


