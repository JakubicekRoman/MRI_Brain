# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:43:00 2022

@author: jakubicek
"""


import sys
import numpy as np
import numpy.matlib
import os
import shutil
import random
import pydicom as dcm
import SimpleITK as sitk   
import matplotlib.pyplot as plt

import Utilities as util


## reference
data_directory_ref = 'D:\Projekty\BrainFNUSA\MRI_gliom\Python\\atlastImage.nii.gz'
# data_directory_ref = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\T1ce'
# data_directory_ref = 'D:\\Projekty\\BrainFNUSA\\MRI_gliom\\Python\\Outputs\\Ambroz_T1ce_0_reg.nii.gz'


# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\T1ce'
# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\T1'
# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\T2'
# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\FLAIR'
data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\DTI'
# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\DTI_01t'
# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\DCE'
# data_directory = 'D:\Projekty\BrainFNUSA\MRI_gliom\data_roman\Ambrozek\DSC'


out_dir = 'D:\Projekty\BrainFNUSA\MRI_gliom\Python\Outputs'
out_dir_temp = out_dir + '\\temp'

if os.path.exists(out_dir_temp):
    shutil.rmtree(out_dir_temp)
os.makedirs(out_dir_temp)


##----- loading ----- resamplig and rotating

name = 'Ambroz'
seq = os.path.basename(data_directory)
ser = 0

if '.nii' in data_directory_ref:
    path_ref = out_dir_temp + '\\' + name + '_' + seq + '_ref.nii.gz'
    shutil.copyfile(data_directory_ref, path_ref)
    info_ref = util.read_nii_info(data_directory_ref)

else:
    info_ref = util.read_dicom_info(data_directory_ref)
    path_ref = util.resave_dicom(data_directory_ref, out_dir_temp, name + '_' + seq + '_ref', 0, info_ref)

info_mov = util.read_dicom_info(data_directory)
    
path_mov = util.resave_dicom(data_directory, out_dir_temp, name + '_' + seq + '_mov', ser, info_mov)


# info = util.read_nii_info(path_mov)





# ##----- registeration ----- 

# ## first time point

# # parameters
# nIter = '100x100x50'
# # ia = ''
# ia = '-ia-image-centers'

# # finding the transf matrix 
# os.system(r'C:/CaPTk_Full/1.9.0/bin/greedy.exe' + ' -d 3 -a -ri LINEAR -dof 6 -m NMI -n ' + nIter + ' ' + ia + ' -i ' 
#           + path_ref + ' ' + path_mov + ' -o ' + out_dir_temp  + '\\transf_matrix.mat')

# # geom transformation
# path_save_reg = out_dir + '\\' + name + '_' + seq + '_' + str(ser) + '_reg.nii.gz'
# os.system( r'C:/CaPTk_Full/1.9.0/bin/greedy.exe -d 3 -r ' + out_dir_temp + '\\transf_matrix.mat -rf ' + path_ref + ' -rm ' + path_mov + ' ' + path_save_reg + ' -ri LINEAR' ) 





# ##----- display results of reg ----- 
# sl = 0.6
# img_ref = util.read_nii(path_ref, sl)
# img_mov = util.read_nii(path_mov, sl)
# img_reg = util.read_nii(path_save_reg, sl)

# img_ref = (( img_ref - img_ref.min() ) / (img_ref.max() - img_ref.min()) )
# img_mov = (( img_mov - img_mov.min() ) / (img_mov.max() - img_mov.min()) )
# img_reg = (( img_reg - img_reg.min() ) / (img_reg.max() - img_reg.min()) )


# img_mov = util.resize_with_padding(img_mov, img_ref.shape )

# fig, axs = plt.subplots(2,2)
# axs[0,0].imshow(img_ref, cmap='gray')
# axs[0,1].imshow(img_mov, cmap='gray')
# axs[1,0].imshow(img_ref, cmap='gray')
# axs[1,1].imshow(img_reg, cmap='gray')

# img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
# img[:,:,0] = img_mov
# img[:,:,1] = img_ref
# img[:,:,2] = img_mov

# plt.figure()
# plt.imshow(img)

# img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
# img[:,:,0] = img_reg
# img[:,:,1] = img_ref
# img[:,:,2] = img_reg

# plt.figure()
# plt.imshow(img)



# # plt.figure()
# # plt.imshow(img_ref, cmap='gray')
# # plt.imshow(img_mov, cmap='jet', alpha=0.2)

# # plt.figure()
# # plt.imshow(img_ref, cmap='gray')
# # plt.imshow(img_reg, cmap='jet', alpha=0.2)



# ## ---------- Skull striping ---------------

# path_data =  ( out_dir + '\\' + name + '_T1_0_reg.nii.gz,' +
#               out_dir + '\\' + name + '_T1ce_0_reg.nii.gz,' +
#               out_dir + '\\' + name + '_T2_0_reg.nii.gz,' +
#               out_dir + '\\' + name + '_FLAIR_0_reg.nii.gz' )

# os.system( r'C:/CaPTk_Full/1.9.0/bin/deepMedic.exe ' + '-md C:/CaPTk_Full/1.9.0/data/deepMedic/saved_models/skullStripping -i '
#           + path_data + ' -o ' + out_dir_temp + '/SkullMask.nii.gz')


    
# ## ---------- Tumour segmentation ---------------

# os.system( r'C:/CaPTk_Full/1.9.0/bin/deepMedic.exe ' + '-md C:/CaPTk_Full/1.9.0/data/deepMedic/saved_models/brainTumorSegmentation -i '
#           + path_data + ' -m ' + out_dir_temp + '/SkullMask.nii.gz'  +' -o ' + out_dir_temp + '/outputSegmentation.nii.gz')




