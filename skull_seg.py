# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:20:54 2022

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
import nibabel as nib

import Utilities as util



pathDir = 'C:\Data\Jakubicek\MRI_Brain\Ambrozek\Outputs'
# pathDir = 'C:\Data\Jakubicek\MRI_Brain\Bednarova\Outputs'
# pathDir = 'C:\Data\Jakubicek\MRI_Brain\Cip\Outputs'

D = [ f for f in os.listdir(pathDir) if os.path.isfile(pathDir+'\\'+f) if (pathDir+'\\'+f).__contains__('.nii') ]


mask_Brain = [ f for f in D if (pathDir+'\\'+f).__contains__('mask_Brain') ]
mask_Tumor = [ f for f in D if (pathDir+'\\'+f).__contains__('mask_Tumor') ]

T1 = [ f for f in D if (pathDir+'\\'+f).__contains__('T1_') ]
T1ce = [ f for f in D if (pathDir+'\\'+f).__contains__('T1ce') ]
T2 = [ f for f in D if (pathDir+'\\'+f).__contains__('T2') ]
Flair = [ f for f in D if (pathDir+'\\'+f).__contains__('FLAIR') ]

DTI = [ f for f in D if (pathDir+'\\'+f).__contains__('DTI') ]
DCE = [ f for f in D if (pathDir+'\\'+f).__contains__('DCE') ]
DSC = [ f for f in D if (pathDir+'\\'+f).__contains__('DSC') ]

# files =  T1 + T1ce + T2 + Flair + DTI + DCE + DSC
files = DCE
name_file = 'DCE_skull'


out_dir = pathDir + '\\comp'

# os.mkdir(pathDir)
# os.mkdir(path_save)


## save mask to 4D image
# img[:,:,:,0] = mask_filt
img=[]

## save other images to 4D image
for i,file in enumerate(files):
    path_data =  ( pathDir + '\\' + file )

    os.system( r'C:\\CaPTk_Full\\1.9.0\\bin\\deepMedic.exe ' + '-md C:\\CaPTk_Full\\1.9.0\\data\\deepMedic\\saved_models\\skullStripping -i '
              + path_data + ' -o ' + out_dir + '\\Skull\\SkullMask.nii.gz')
    
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("NiftiImageIO")
    file_reader.SetFileName(out_dir + '\\Skull\\SkullMask.nii.gz')
    img1 = file_reader.Execute()
    img1 = sitk.GetArrayFromImage(img1).astype(np.uint8)
    img1 = np.transpose(img1,(2,1,0))
    # img[:,:,:,i+1] = img1[BB[0][0]:BB[0][1], BB[1][0]:BB[1][1], BB[2][0]:BB[2][1]]
    img[:,:,:,i+1] = img1


# img = np.transpose(img,(3,2,1,0))

# for x in range(0,img.shape[0]):
#     for y in range(0,img.shape[1]):
#         for z in range(0,img.shape[2]):
#             for t in range(0,img.shape[3]):
#                 # sImg[x,y,z,:] = tuple([int(row) for row in img[x,y,z,:]])
#                 sImg[x,y,z,t] = int(img[x,y,z,t])
              
   


path_save_cur = out_dir + '\\' + name_file + '.nii.gz'
# data = np.ones((32, 32, 15, 100), dtype=np.int16) # dummy data in numpy matrix
data = nib.Nifti1Image(img, np.eye(4))  # Save axis for data (just identity)

data.header.get_xyzt_units()
data.to_filename(path_save_cur)  # Save as NiBabel file



# img = tuple([int(row) for row in np.concatenate(np.concatenate(np.concatenate(img)))])
# sImg[:,:,:,:] = img[:,:,:,:]

# img4 = sitk.GetImageFromArray(img)

# joiner = sitk.JoinSeriesImageFilter()
# img3 = joiner.Execute(img1, img2)
# img3_arr = sitk.GetArrayFromImage(img3)

# joiner = sitk.JoinSeriesImageFilter()
# img3 = joiner.Execute( sitk.GetImageFromArray(img[0,:,:,:]), sitk.GetImageFromArray(img[1,:,:,:]), sitk.GetImageFromArray(img[2,:,:,:]))


# img3_arr = sitk.GetArrayFromImage(img3)
# img3_arr = np.transpose(img3_arr,(3,2,1,0))
# img3_rec = sitk.GetImageFromArray(img3_arr)


# ## saving
# writer = sitk.ImageFileWriter()
# path_save_cur = path_save + '\\test.nii.gz'
# writer = sitk.ImageFileWriter()
# writer.SetFileName(path_save_cur)
# writer.Execute(Img)

# writer = sitk.ImageFileWriter()
# path_save_cur = path_save + '\\test7.nii.gz'
# writer = sitk.ImageFileWriter()
# writer.SetFileName(path_save_cur)
# writer.Execute(sImg)


text_file = open( path_save + '\\' + name_file + 'cropping.txt', "w")
params = 'x = {},{}\n'.format(BB[0][0],BB[0][1]) 
text_file.write(params)
params = 'y = {},{}\n'.format(BB[1][0],BB[1][1]) 
text_file.write(params)
params = 'z = {},{}'.format(BB[2][0],BB[2][1]) 
text_file.write(params)
text_file.close()



text_file = open( path_save + '\\' + name_file + 'stack_maps.txt', "w")
indexS = 1 + 1
indexE = 1 + len(T1)
params =  'T1 - num volumes: '+ str(len(T1)) + ', range: '+ str(indexS)+':'+str(indexE)  +'\n'
text_file.write(params)
indexS = indexE + 1
indexE = indexE + len(T1ce)
params =  'T1ce - num volumes: '+ str(len(T1ce)) +', range: '+ str(indexS)+':'+str(indexE)  +'\n' 
text_file.write(params)
indexS = indexE + 1
indexE = indexE + len(T2)
params =  'T2 - num volumes: '+ str(len(T2)) +', range: '+ str(indexS)+':'+str(indexE)  +'\n' 
text_file.write(params)
indexS = indexE + 1
indexE = indexE + len(Flair)
params =  'FLAIR - num volumes: '+ str(len(Flair)) +', range: '+ str(indexS)+':'+str(indexE)  +'\n' 
text_file.write(params)
indexS = indexE + 1
indexE = indexE + len(DTI)
params =  'DTI - num volumes: '+ str(len(DTI)) +', range: '+ str(indexS)+':'+str(indexE)  +'\n' 
text_file.write(params)
indexS = indexE + 1
indexE = indexE + len(DCE)
params =  'DCE - num volumes: '+ str(len(DCE)) +', range: '+ str(indexS)+':'+str(indexE)  +'\n' 
text_file.write(params)
indexS = indexE + 1
indexE = indexE + len(DSC)
params =  'DSC - num volumes: '+ str(len(DSC)) +', range: '+ str(indexS)+':'+str(indexE)
text_file.write(params)
text_file.close()









