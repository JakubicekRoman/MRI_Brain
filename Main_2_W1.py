# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:44:37 2022

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



pathDir = 'C:\Data\Jakubicek\MRI_Brain\Ambrozek\Outputs'

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

files =  T1 + T1ce + T2 + Flair + DTI + DCE + DSC

path_save = pathDir + '\\comp'
# os.mkdir(path_save)


## loading of images
i = 0
file_reader = sitk.ImageFileReader()
file_reader.SetImageIO("NiftiImageIO")
file_reader.SetFileName(pathDir + '\\' + files[i])
img1 = file_reader.Execute()

i = 100
file_reader = sitk.ImageFileReader()
file_reader.SetImageIO("NiftiImageIO")
file_reader.SetFileName(pathDir + '\\' + files[i])
img2 = file_reader.Execute()

joiner = sitk.JoinSeriesImageFilter()
img3 = joiner.Execute(img1, img2)


## finding Borders of BB mask
file_reader = sitk.ImageFileReader()
file_reader.SetImageIO("NiftiImageIO")
file_reader.SetFileName(pathDir + '\\' + mask_Tumor[0])
file_reader.ReadImageInformation()
vel = file_reader.GetSize()
maskSitk = file_reader.Execute()
mask = sitk.GetArrayFromImage(file_reader.Execute())

BB = util.bound_3D(mask, 0)

## cropping
# img4 = img3[BB[4]:BB[5], BB[2]:BB[3], BB[0]:BB[1], 0:2]
img4 = maskSitk[BB[2][0]:BB[2][1], BB[1][0]:BB[1][1], BB[0][0]:BB[0][1]]


## saving
writer = sitk.ImageFileWriter()   
path_save_cur = path_save + '\\test.nii.gz'
writer = sitk.ImageFileWriter()    
writer.SetFileName(path_save_cur)
writer.Execute(img4)












