import os
import shutil

if not os.path.exists('../data/full_images'):
    os.makedirs('../data/full_images')

PATH = "../data/_DataSet Cervical Cells/"

i = 1
for dir in os.listdir(PATH):
    dir_ = os.path.join(PATH, dir)
    for f in os.listdir(dir_):
        print(f)
        shutil.copy(os.path.join(dir_, f),
                    '../data/full_images/{}_{}.jpg'.format(i, dir))
        i += 1