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
import copy

import Utilities as util


# pat = 'C:\Data\Jakubicek\MRI_Brain\Ambrozek'
# pat = 'C:\Data\Jakubicek\MRI_Brain\Bednarova'
pat = 'C:\Data\Jakubicek\MRI_Brain\Cip'

## reference
# data_directory_ref = 'D:\Projekty\BrainFNUSA\MRI_gliom\Python\\atlastImage.nii.gz'
# data_directory_ref = pat + '\T1ce'
# data_directory_ref = 'D:\\Projekty\\BrainFNUSA\\MRI_gliom\\Python\\Outputs\\Ambroz_T1ce_0_reg.nii.gz'


data_directory = (pat)

name = os.path.basename(data_directory)

out_dir =  pat + '\\Outputs'

out_dir_temp = out_dir + '\\temp'

if os.path.exists(out_dir_temp):
    shutil.rmtree(out_dir_temp)
os.makedirs(out_dir_temp)


### ----- loading ----- resamplig and rotating, registrace

seqs = ( 'T1', 'T2', 'FLAIR', 'DTI', 'DCE', 'DSC')
# seqs = ( 'DSC',  )

### ----- pre-registration -----

path_ref = os.getcwd() + '\\atlastImage.nii.gz'
seq = 'T1ce'
ser = 0

data_directory_ref = pat + '\\' + seq
info_ref = util.read_dicom_info(data_directory_ref)
path_mov = util.resave_dicom(data_directory_ref, out_dir_temp, 'mov_'+seq, 0, info_ref, bias=True)

    
# parameters
nIter = '100x100x50'
# ia = ''
ia = '-ia-image-centers'

# finding the transf matrix 
transfPre = 'transf_matrix_preReg_' + seq + '.mat'
os.system( os.getcwd() + '\\greedy.exe' + ' -d 3 -a -ri LINEAR -dof 6 -m NMI -n ' + nIter + ' ' + ia + ' -i ' 
          + path_ref + ' ' + path_mov + ' -o ' + out_dir_temp  + '\\' + transfPre )

# geom transformation
path_save_reg = out_dir + '\\' + name+'_'+seq + '_' + str(ser) + '_reg.nii.gz'
os.system( os.getcwd() + '\\greedy.exe -d 3 -r ' + out_dir_temp + '\\' + transfPre + ' -rf ' + path_ref + ' -rm ' + path_mov + ' ' + path_save_reg + ' -ri LINEAR' ) 



## display of registration
sl=0.5
util.display_reg(path_ref,path_mov,path_save_reg, sl)



###### registration of other moving images

path_ref = path_save_reg

for i, seq in enumerate(seqs): 
    # seq = 'FLAIR'
    ser = 0  
    info_mov = util.read_dicom_info(data_directory + '\\' + seq)
    path_mov = util.resave_dicom(data_directory + '\\' + seq, out_dir_temp, 'mov_'+seq , ser, info_mov, bias=False)
    path_movO = copy.deepcopy(path_mov)

    #----- registeration -----
    
    # first time point
    
    # parameters
    nIter = '100x100x50'
    ia = ''
    # ia = '-ia-image-centers'
    ia = '-ia ' + out_dir_temp + '\\' + transfPre
    it = ''
    # it = ' -it ' + out_dir_temp + '\\' + transfPre
    
    # finding the transf matrix 
    transf = 'transf_matrix_' + seq + '.mat'
    # os.system( os.getcwd() + '\\greedy.exe' + ' -d 3 -a -ri LINEAR -dof 6 -m NMI -n ' + nIter + ' ' + ia + ' -i '
    #           + path_ref + ' ' + path_mov + ' -o ' + out_dir_temp  + '\\' + transf )
    os.system( os.getcwd() + '\\greedy.exe' + ' -d 3 -a -ri LINEAR -dof 6 -m NMI -n ' + nIter + ' ' + ia + it + ' -i '
              + path_ref + ' ' + path_mov + ' -o ' + out_dir_temp  + '\\' + transf )
        
    # geom transformation for each time poitn/direction
    
    for ser in range(0,int(info_mov['series']),1):
    # for ser in range(0,60,1):
        path_mov = util.resave_dicom(data_directory + '\\' + seq, out_dir_temp, 'mov_'+seq , ser, info_mov, bias=False)
        
        path_save_reg = out_dir + '\\' + name+'_'+seq + '_' + str(ser) + '_reg.nii.gz'
        os.system( os.getcwd() + '\\greedy.exe -d 3 -r ' + out_dir_temp + '\\' + transf + ' -rf ' + path_ref + ' -rm ' + path_mov + ' ' + path_save_reg + ' -ri LINEAR' ) 
    
        ## display of registration
        # sl=0.35
        # util.display_reg(path_ref,path_mov,path_save_reg, sl)

    ser = 0
    path_save_reg = out_dir + '\\' + name+'_'+seq + '_' + str(ser) + '_reg.nii.gz'
    sl=0.5
    util.display_reg(path_ref,path_movO,path_save_reg, sl)


##### NEXT PART .... SEGMENTATION
   
# ## ---------- Skull striping ---------------

path_data =  ( out_dir + '\\' + name + '_T1_0_reg.nii.gz,' +
              out_dir + '\\' + name + '_T1ce_0_reg.nii.gz,' +
              out_dir + '\\' + name + '_T2_0_reg.nii.gz,' +
              out_dir + '\\' + name + '_FLAIR_0_reg.nii.gz' )

os.system( r'C:\\CaPTk_Full\\1.9.0\\bin\\deepMedic.exe ' + '-md C:\\CaPTk_Full\\1.9.0\\data\\deepMedic\\saved_models\\skullStripping -i '
          + path_data + ' -o ' + out_dir_temp + '\\Skull\\SkullMask.nii.gz')


# ## ---------- Masking registered data --☻-------------

# mask = util.read_nii(  out_dir_temp + '\\Skull\\SkullMask.nii.gz' , -1)
file_reader = sitk.ImageFileReader()
file_reader.SetImageIO("NiftiImageIO")
file_reader.SetFileName(out_dir_temp + '\\Skull\\SkullMask.nii.gz')
mask = file_reader.Execute()
    
path_data2 = path_data.split(',')

path_data_masked = ('')
for i in range(0,len(path_data2)):
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("NiftiImageIO")
    file_reader.SetFileName(path_data2[i])
    img = file_reader.Execute()
    img_masked = sitk.Mask(  sitk.Cast( img, sitk.sitkInt16), sitk.Cast(mask, sitk.sitkInt16)  )
    writer = sitk.ImageFileWriter()   
    path_save = path_data2[i].replace('_reg','_reg_masked')
    path_save = path_save.replace('Outputs\\','Outputs\\temp\\')
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(path_save)
    writer.Execute(img_masked)
    
    path_data_masked = path_data_masked + path_save + ','


path_data_masked = path_data_masked[:-1]
# plt.figure()
# plt.imshow(img[:,:,80])

    
## ---------- Tumour segmentation ---------------

os.system( r'C:\\CaPTk_Full\\1.9.0\\bin\\deepMedic.exe ' + '-md C:\\CaPTk_Full\\1.9.0\\data\\deepMedic\\saved_models\\brainTumorSegmentation -i '
          + path_data_masked + ' -m ' + out_dir_temp + '\\Skull\\SkullMask.nii.gz'  +' -o ' + out_dir_temp + '\\Tumour\\outputSegmentation.nii.gz')

shutil.copyfile(out_dir_temp + '\\Tumour\\outputSegmentation.nii.gz', out_dir + '\\' + name + '_mask_Tumor.nii.gz' )
shutil.copyfile(out_dir_temp + '\\Skull\\SkullMask.nii.gz', out_dir + '\\' + name + '_mask_Brain.nii.gz' )


# if os.path.exists(out_dir_temp):
#     shutil.rmtree(out_dir_temp)
