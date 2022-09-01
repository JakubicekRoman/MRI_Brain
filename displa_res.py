# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:00:48 2022

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


# path_ref = 'D:\Projekty\BrainFNUSA\MRI_gliom\Python\Outputs\Ambrozek\temp\fix_T1ce_0.nii.gz'
# path_mov = 'D:\Projekty\BrainFNUSA\MRI_gliom\Python\Outputs\Ambrozek\reg_T1_0.nii.gz'
# path_save_reg = 'D:\Projekty\BrainFNUSA\MRI_gliom\Python\Outputs\Ambrozek\temp\mov_T1_0.nii'


def display_reg(path_ref,path_mov,path_save_reg, sl):
    
    ##----- display results of reg ----- 
    # sl = 0.6
    img_ref = util.read_nii(path_ref, sl)
    img_mov = util.read_nii(path_mov, sl)
    img_reg = util.read_nii(path_save_reg, sl)
    
    img_ref = (( img_ref - img_ref.min() ) / (img_ref.max() - img_ref.min()) )
    img_mov = (( img_mov - img_mov.min() ) / (img_mov.max() - img_mov.min()) )
    img_reg = (( img_reg - img_reg.min() ) / (img_reg.max() - img_reg.min()) )
    
    
    img_mov = util.resize_with_padding(img_mov, img_ref.shape )
    
    # fig, axs = plt.subplots(2,2)
    # axs[0,0].imshow(img_ref, cmap='gray')
    # axs[0,1].imshow(img_mov, cmap='gray')
    # axs[1,0].imshow(img_ref, cmap='gray')
    # axs[1,1].imshow(img_reg, cmap='gray')
    
    img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
    img[:,:,0] = img_mov
    img[:,:,1] = img_ref
    img[:,:,2] = img_mov
    
    # plt.figure()
    # plt.imshow(img)
    
    fig, axs = plt.subplots(2,2)        
    axs[0,0].imshow(img)
    
    img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
    img[:,:,0] = img_reg
    img[:,:,1] = img_ref
    img[:,:,2] = img_reg
    
   
    # plt.figure()
    # plt.imshow(img)

    axs[0,1].imshow(img)
    
   
    # plt.figure()
    # plt.imshow(img_ref, cmap='gray')
    # plt.imshow(img_mov, cmap='jet', alpha=0.2)
    
    # plt.figure()
    # plt.imshow(img_ref, cmap='gray')
    # plt.imshow(img_reg, cmap='jet', alpha=0.2)
