# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:55:08 2023

@author: jakubicek
"""

# import sys
import numpy as np
# import numpy.matlib
import os
import shutil
import glob
# import random
# import pydicom as dcm
# import SimpleITK as sitk   
# import matplotlib.pyplot as plt
# import copy
# import nibabel as nib

# import Utilities as util

# import nipype.interfaces.dcm2nii as Dcm2nii
# import nipype.interfaces.dcm2nii
# from nipype.interfaces.dcm2nii import Dcm2niix
# from nipype.interfaces import niftyfit

import dcm2niix
import napari

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti
from dipy.data import get_fnames

from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa

import nibabel as nib

# import Utilities as util



#%% 
data_directory = 'C:\Data\MRI_Glioms\MRI_Brain\Bogner'

name = os.path.basename(data_directory)
out_dir =  data_directory + '\\Outputs_maps\\DTI'

# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
# os.makedirs(out_dir)

# out_dir_temp = out_dir + '\\temp\\DTI'

# if os.path.exists(out_dir_temp):
#     shutil.rmtree(out_dir_temp)
# os.makedirs(out_dir_temp)

file_path = data_directory + '\\DTI'

cmd = 'dcm2niix -b y -z y -x n -t n -m n -o ' + out_dir + ' -s n -v n ' + file_path
# os.system(cmd)

# data_fname = glob.glob(out_dir + '\\*.nii.gz')[0]
bval_fname = glob.glob(out_dir + '\\*.bval')[0]
bvec_fname = glob.glob(out_dir + '\\*.bvec')[0]


data_fname = glob.glob(data_directory + '\\Outputs\\*DTI.nii.gz')[0]

mask_fname = glob.glob(data_directory + '\\Outputs\\*mask*.nii.gz')[0]



#%%  loading

# hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
# data, affine, coord = load_nifti(hardi_fname, return_img=False,return_coords=True)
# bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)

# data, affine = load_nifti(data_fname)
data, affine, coord = load_nifti(data_fname, return_img=False,return_coords=True)

# affine[0,3] = 0
# affine[1,3] = 0
# affine[2,3] = 0

bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)

# data = util.check_orientation(data, coord)
# new_nifti = nib.Nifti1Image(data, affine)
# nib.save(new_nifti, f'' + data_fname.replace('.nii','_RPS.nii') )
# data, affine, coord = load_nifti(data_fname.replace('.nii','_RPS.nii'), return_img=False,return_coords=True)


gtab = gradient_table(bvals, bvecs)


mask, affine, coord = load_nifti(mask_fname, return_img=False,return_coords=True)

maskdata=np.zeros_like(data)
for i in range(data.shape[3]):
    maskdata[...,i] = data[...,i]*mask

# maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
#                              numpass=1, autocrop=True, dilate=2)
# print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

# maskdata, mask = median_otsu(data, vol_idx=range(5, 34), median_radius=3,
#                              numpass=1, autocrop=True, dilate=2)
# print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)


# # data = np.ndarray.transpose(data,[2,3,1,0])
# viewer = napari.view_image(np.ndarray.transpose(data,[2,3,1,0]))
# viewer = napari.view_image(data)

# viewer = napari.view_image(np.ndarray.transpose(maskdata,[2,3,1,0]))
# # viewer = napari.view_image(maskdata)

# save_nifti('new_dataM.nii.gz', data.astype(np.float32), affine)


#%% to FA map

tenmodel = dti.TensorModel(gtab)

tenfit = tenmodel.fit(maskdata)

tensor_vals = dti.lower_triangular(tenfit.quadratic_form)

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

# FA = np.ndarray.transpose(FA,[2,1,0])
# viewer = napari.view_image(FA)

# save_nifti(data_directory + '\\Outputs\\' + 'FA_niix.nii.gz', FA.astype(np.float32), affine)
save_nifti(data_directory + '\\Outputs\\' + 'FA_sitk.nii.gz', FA.astype(np.float32), affine)

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
save_nifti(data_directory + '\\Outputs\\' + 'FA_RGB_sitk.nii.gz', np.array(255 * RGB, 'uint8'), affine)


#%% ADC map

MD = dti.mean_diffusivity(tenfit.evals)
MD[np.isnan(MD)] = 0

# MD1 = np.ndarray.transpose(MD,[2,1,0])
# viewer = napari.view_image(MD1)

# save_nifti(data_directory + '\\Outputs\\' + 'MD_niix.nii.gz', MD.astype(np.float32), affine)
save_nifti(data_directory + '\\Outputs\\' + 'MD_sitk.nii.gz', MD.astype(np.float32), affine)





