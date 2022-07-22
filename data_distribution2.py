import torch
import pandas as pd
import os
from glob import glob
import shutil

index = 0
all_xrays_df = pd.read_csv('./data_list/Pneumothorax.csv') # 

# folder path 
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

pathes = [Atelectasis_path, Cardiomegaly_path, Consolidation_path, Edema_path, Effusion_path, Emphysema_path, Fibrosis_path, Hernia_path, Infiltration_path, Mass_path, Nodule_path, Pleural_Thickening_path, Pneumothorax_path, Pneumonia_path, Nofinding_path]
diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumothorax', 'Pneumonia', 'NoFinding']


for i in range(len(pathes)):
    index = 0
    all_xrays_df = pd.read_csv('./data_list/' + diseases[i] + '.csv')
    print(diseases[i])
    for row in all_xrays_df.itertuples():
        src = row[14]
        img_name =  diseases[i] + '_' + str(index) + '.png' # 
        dst = pathes[i] + img_name # 
        shutil.copy(src,dst)
        index += 1

# for row in all_xrays_df.itertuples():
#     src = row[14]
#     img_name = 'NoFinding_' + str(index) + '.png' # 
#     dst = Nofinding_path + img_name # 
#     shutil.copy(src,dst)
#     index += 1
        








