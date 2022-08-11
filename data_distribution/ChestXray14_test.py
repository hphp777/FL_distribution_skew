import torch
import pandas as pd
import os
from glob import glob
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt

all_xrays_df = pd.read_csv('./Data_Entry.csv')

my_file = open("test_list.txt", "r")
data = my_file.read()
testdata = data.replace('\n', ' ').split(" ")

print(len(testdata))

all_image_paths = {os.path.basename(x): x for x in 
                   glob('C:/Users/hb/Desktop/Data/archive/images_*/images/*.png')}
all_xrays_df['FilePath'] = all_xrays_df['Image Index'].map(all_image_paths.get)

index = 0

for row in all_xrays_df.itertuples():
    if row[1] in testdata:
        src = row[13]
        dst = 'C:/Users/hb/Desktop/data/Chest-Xray14/test/' + row[2].replace(' ','').split('|')[0] + '_' + str(index) + '.png'
        # shutil.copy(src,dst)
        os.remove(src)
        index += 1