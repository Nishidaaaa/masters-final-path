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


index_path = './dataset/nikon/index.xlsx' #
filepaths = glob.glob("./dataset/nikon/**/**/*.jpg")   
storepath = './dataset/CTS/' #データセットを保存するフォルダ 
classes = {'ca','cang','ng','nr1','nr2'}
magnification = '200x'


if os.path.isdir(storepath):
    shutil.rmtree(storepath)   
os.mkdir(storepath)

sheet = pd.read_excel(index_path, sheet_name=0, usecols=["検体番号", "永久ＨＥnear1(癌1,正常0）", "永久ＨＥnr2(癌1,正常0）"])
sheet = sheet.rename(columns={"検体番号": "number", "永久ＨＥnear1(癌1,正常0）": "nr1", "永久ＨＥnr2(癌1,正常0）": "nr2"})
sheet = sheet.dropna(axis=0)
sheet = sheet.astype({"number": str, "nr1": bool, "nr2": bool})
        
for filepath in filepaths:
    filename = os.path.basename(filepath) #パスからファイル名を取得.
    root, ext = os.path.splitext(filename)
    m = re.match(r'([a-z]+)\.([a-zA-Z0-9]+)\.([0-9]+)\.([a-zA-Z0-9]+)', filename)
    
   
    if m is not None: # 対象ファイル
        newfilename = ''
        if(m[4] == magnification) :
            
            if m[2] == 'ca':
                shutil.copy(filepath, storepath)
                newfilename = m[0] + "_1" + ext
                os.rename(storepath + filename,storepath + newfilename)
            
            if m[2] == 'cang':
                shutil.copy(filepath, storepath)
                newfilename = m[0] + "_1" + ext
                os.rename(storepath + filename,storepath + newfilename)
            
            if m[2] == 'ng':
                shutil.copy(filepath, storepath)
                newfilename = m[0] + "_0" + ext
                os.rename(storepath + filename,storepath + newfilename)
            
            if m[2] == 'nr1': 
                for i in range(sheet.shape[0]) :
                    if m[3] == sheet.at[i,"number"] :
                        shutil.copy(filepath, storepath)
                        label = int(sheet.at[i,"nr1"]) 
                        newfilename = m[0] + "_" + str(label) + ext
                        os.rename(storepath + filename,storepath + newfilename)

            if m[2] == 'nr2':      
                for i in range(sheet.shape[0]) :
                    if m[3] == sheet.at[i,"number"] :
                        shutil.copy(filepath, storepath)
                        label = int(sheet.at[i,"nr2"]) 
                        newfilename = m[0] + "_" + str(label) + ext
                        os.rename(storepath + filename,storepath + newfilename)
            #else :
                #print(filename + ' --> ')               
        print(filename + ' --> ' + newfilename)        

random_sample_file(storepath) #画像ファイルをTRAINとTESTに振り分ける


