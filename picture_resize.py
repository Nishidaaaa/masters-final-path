import numpy as np
from PIL import Image
import glob
import os
import shutil
import re
import random
import openpyxl
import pandas as pd
import math

def random_sample_file(storepath):
    SAMPLING_RATIO = 0.3

    files = glob.glob(storepath + '*')
    random_sample_file = random.sample(files,math.ceil(len(files)*SAMPLING_RATIO))
    os.mkdir(storepath + 'TRAIN')
    os.mkdir(storepath + 'TEST')

    for file in random_sample_file:
        shutil.move(file, storepath + 'TEST')

    files = glob.glob(storepath + '*.png')
    for file in  files :
        shutil.move(file, storepath + 'TRAIN')

filepaths = glob.glob("./dataset/PathMNIST/Allpathmnist/**/**/*.png")   
storepath = './dataset/Path_all_x252/' #データセットを保存するフォルダ 

if os.path.isdir(storepath):
    shutil.rmtree(storepath)   
os.mkdir(storepath)

# 各CSVファイルを読み込み、Excel形式で出力
for filepath in filepaths:
    shutil.copy(filepath, storepath)
    
filepaths = glob.glob("./dataset/Path_all_x252/*.png")   

for filepath in filepaths:
    # CSVファイルを読み込む
    image = np.asarray(Image.open(filepath).convert("RGB"), dtype=np.uint8)
    zoomed_image = image.repeat(9, axis=0).repeat(9, axis=1)
    Image.fromarray(zoomed_image).save(filepath)

random_sample_file(storepath)

