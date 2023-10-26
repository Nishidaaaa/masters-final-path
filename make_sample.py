import shutil
import glob
import os
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

    files = glob.glob(storepath + '*.jpg')
    for file in  files :
        shutil.move(file, storepath + 'TRAIN')

def take_sample_file(datapath,storepath):
    SAMPLING_RATIO = 0.5

    files = glob.glob(datapath + '**/*.jpg')
    random_sample_file = random.sample(files,math.ceil(len(files)*SAMPLING_RATIO))
    
    for file in random_sample_file:
        shutil.copy(file, storepath)


datapath = './dataset/CTS/'
storepath = './dataset/CTS_sample_150/' #データセットを保存するフォルダ 

os.mkdir(storepath)
take_sample_file(datapath,storepath)
random_sample_file(storepath) #画像ファイルをTRAINとTESTに振り分ける