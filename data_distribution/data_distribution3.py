import torch
import pandas as pd
import os
from glob import glob
import shutil
import math

# Label skew for each client

Atelectasis_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Atelectasis/'
Cardiomegaly_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Cardiomegaly/'
Consolidation_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Consolidation/'
Edema_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Edema/'
Effusion_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Effusion/'
Emphysema_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Emphysema/'
Fibrosis_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Fibrosis/'
Hernia_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Hernia/'
Infiltration_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Infiltration/'
Mass_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Mass/'
Nodule_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Nodule/'
Pleural_Thickening_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Pleural_Thickening/'
Pneumothorax_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Pneumothorax/'
Pneumonia_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/Pneumonia/'
Nofinding_path = 'C:/Users/hb/Desktop/Data/ChestX-ray14/NoFinding/'

Atelectasis_ratio = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
Cardiomegaly_ratio = [0.1,0.3,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.2]
Consolidation_ratio = [0.05,0.2,0.05,0.1,0.1,0.1,0.1,0,0.1,0.2]
Edema_ratio = [0.05,0.2,0.05,0.1,0.1,0.1,0.1,0,0.1,0.2]
Effusion_ratio = [0.1,0.3,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.2]
Emphysema_ratio = [0.05,0.2,0.05,0.1,0.1,0.1,0.1,0,0.1,0.2]
Fibrosis_ratio = [0.1,0.2,0.1,0.1,0.1,0.1,0.1,0,0.1,0.1]
Hernia_ratio = [0.1,0.3,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.2]
Infiltration_ratio = [0.05,0.2,0.05,0.1,0.1,0.1,0.1,0,0.1,0.2]
Mass_ratio = [0.05,0.2,0.05,0.1,0.1,0.1,0.1,0,0.1,0.2]
Nodule_ratio = [0.1,0.3,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.2]
Pleural_Thickening_ratio = [0.05,0.2,0.05,0.1,0.1,0.1,0.1,0,0.1,0.2]
Pneumothorax_ratio = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
Pneumonia_ratio = [0.1,0.3,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.2]
Nofinding_ratio = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

pathes = [Atelectasis_path, Cardiomegaly_path, Consolidation_path, Edema_path, Effusion_path, Emphysema_path, Fibrosis_path, Hernia_path, Infiltration_path, Mass_path, Nodule_path, Pleural_Thickening_path, Pneumothorax_path, Pneumonia_path, Nofinding_path]
diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumothorax', 'Pneumonia', 'NoFinding']
ratios = [Atelectasis_ratio, Cardiomegaly_ratio, Consolidation_ratio, Edema_ratio, Effusion_ratio, Emphysema_ratio, Fibrosis_ratio, Hernia_ratio, Infiltration_ratio, Mass_ratio, Nodule_ratio, Pleural_Thickening_ratio, Pneumothorax_ratio, Pneumonia_ratio, Nofinding_ratio]

for disease in range(len(pathes)):
    index = 0
    all_image_paths = {os.path.basename(x): x for x in 
                   glob('C:/Users/hb/Desktop/Data/ChestX-ray14/' + diseases[disease] + '/*.png')}
    total_img_num = len(all_image_paths)

    for client in range(10):
        if client == 9:
            for img in range(index, total_img_num):
                img_name = diseases[disease] + '_' + str(img) + '.png'
                src = pathes[disease] + img_name
                dst = 'C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client) + '/' + img_name
                shutil.copy(src,dst)
        else:
            for img in range(index, math.floor(index + ratios[disease][client]*total_img_num)):
                img_name = diseases[disease] + '_' + str(img) + '.png'
                src = pathes[disease] + img_name
                dst = 'C:/Users/hb/Desktop/Data/ChestX-ray14_Client_Data/C' + str(client) + '/' + img_name
                shutil.copy(src,dst)
        index = index + math.floor(ratios[disease][client]*total_img_num)