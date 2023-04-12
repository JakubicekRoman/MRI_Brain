# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:43:00 2022

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
import glob

import Utilities as util

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
import napari

# pat = 'C:\Data\MRI_Glioms\MRI_Brain\Ambrozek'
# pat = 'C:\Data\MRI_Glioms\MRI_Brain\Bednarova'
# pat = 'C:\Data\MRI_Glioms\MRI_Brain\Cip'
pat = 'C:\Data\MRI_Glioms\MRI_Brain\Bogner'

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

seqs = ( 'T1', 'T2', 'FLAIR', 'DTI', 'DCE', 'DSC','Map5','Map10','Map15' )
# seqs = ( 'T1', 'T2', 'FLAIR', 'DTI' )
# seqs = ( 'Map5',  )
# seqs = ( 'T1', 'T2', 'FLAIR','Map5','Map10','Map15' )
# seqs = ( 'DTI',  )

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
path_save_reg = out_dir_temp + '\\' + name+'_'+seq + '_' + f"{ser:04d}" + '_reg.nii.gz'
os.system( os.getcwd() + '\\greedy.exe -d 3 -r ' + out_dir_temp + '\\' + transfPre + ' -rf ' + path_ref + ' -rm ' + path_mov + ' ' + path_save_reg + ' -ri LINEAR' ) 


## display of registration
sl=0.5
util.display_reg(path_ref,path_mov,path_save_reg, sl)

os.remove(path_mov)


###### registration of other moving images

# path_ref = path_save_reg

for i, seq in enumerate(seqs): 
    # seq = 'FLAIR'
    ser = 0  
    info_mov = util.read_dicom_info(data_directory + '\\' + seq)
    path_mov = util.resave_dicom(data_directory + '\\' + seq, out_dir_temp, 'mov_'+seq , ser, info_mov, bias=False)
    path_movO = copy.deepcopy(path_mov)
    
    if 'DTI' in seq or 'DSC' in seq:
        path_ref = out_dir_temp + '\\' + name+'_'+  'T2' + '_' + f"{ser:04d}" + '_reg.nii.gz'
    else:
        path_ref =  out_dir_temp + '\\' + name+'_'+  'T1ce' + '_' + f"{ser:04d}" + '_reg.nii.gz'

    #----- registeration -----
    
    # first time point
    
    # parameters
    nIter = '100x100x100'
    ia = ''
    # ia = '-ia-image-centers'
    ia = '-ia ' + out_dir_temp + '\\' + transfPre
    it = ''
    # it = ' -it ' + out_dir_temp + '\\' + transfPre
    
    
    # finding the transf matrix 
    transf = 'transf_matrix_' + seq + '.mat'
    # os.system( os.getcwd() + '\\greedy.exe' + ' -d 3 -a -ri LINEAR -dof 6 -m NMI -n ' + nIter + ' ' + ia + ' -i '
    #           + path_ref + ' ' + path_mov + ' -o ' + out_dir_temp  + '\\' + transf )
    os.system( os.getcwd() + '\\greedy.exe' + ' -d 3 -a -ri LINEAR -dof 12 -m NMI -n ' + nIter  + ' ' + ia + it + ' -i '
              + path_ref + ' ' + path_mov + ' -o ' + out_dir_temp  + '\\' + transf )
    
    # geom transformation for each time poitn/direction
    
    for ser in range(0,int(info_mov['series']),1):
    # for ser in range(0,60,1):
        path_mov = util.resave_dicom(data_directory + '\\' + seq, out_dir_temp, 'mov_'+seq , ser, info_mov, bias=False)
        
        path_save_reg = out_dir_temp + '\\' + name+'_'+seq + '_' + f"{ser:04d}" + '_reg.nii.gz'
        os.system( os.getcwd() + '\\greedy.exe -d 3 -r ' + out_dir_temp + '\\' + transf + ' -rf ' + path_ref + ' -rm ' + path_mov + ' ' + path_save_reg + ' -ri LINEAR' ) 
    
        ## display of registration
        if ser==0:
            path_save_reg = out_dir_temp + '\\' + name+'_'+seq + '_' + f"{ser:04d}" + '_reg.nii.gz'
            sl=0.5
            util.display_reg(path_ref,path_movO,path_save_reg, sl)
        
        os.remove(path_mov)


##### NEXT PART .... SEGMENTATION
   
# ## ---------- Skull striping ---------------

path_data =  ( out_dir_temp + '\\' + name + '_T1_0000_reg.nii.gz,' +
              out_dir_temp + '\\' + name + '_T1ce_0000_reg.nii.gz,' +
              out_dir_temp + '\\' + name + '_T2_0000_reg.nii.gz,' +
              out_dir_temp + '\\' + name + '_FLAIR_0000_reg.nii.gz' )

os.system( r'C:\\CaPTk_Full\\1.9.0\\bin\\deepMedic.exe ' + '-md C:\\CaPTk_Full\\1.9.0\\data\\deepMedic\\saved_models\\skullStripping -i '
          + path_data + ' -o ' + out_dir_temp + '\\Skull\\SkullMask.nii.gz')


# ## ---------- Masking registered data --------------- ##

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
    # path_save = path_save.replace('Outputs\\','Outputs\\temp\\')
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(path_save)
    writer.Execute(img_masked)
    
    path_data_masked = path_data_masked + path_save + ','


path_data_masked = path_data_masked[:-1]
# plt.figure()
# plt.imshow(img[:,:,80])

    
## ---------- Tumour segmentation --------------- ##

os.system( r'C:\\CaPTk_Full\\1.9.0\\bin\\deepMedic.exe ' + '-md C:\\CaPTk_Full\\1.9.0\\data\\deepMedic\\saved_models\\brainTumorSegmentation -i '
          + path_data_masked + ' -m ' + out_dir_temp + '\\Skull\\SkullMask.nii.gz'  +' -o ' + out_dir_temp + '\\Tumour\\outputSegmentation.nii.gz')



## -----------  COPY to main directiory ------------- ##

shutil.copyfile(out_dir_temp + '\\Tumour\\outputSegmentation.nii.gz', out_dir + '\\' + name + '_mask_Tumor.nii.gz' )
shutil.copyfile(out_dir_temp + '\\Skull\\SkullMask.nii.gz', out_dir + '\\' + name + '_mask_Brain.nii.gz' )

shutil.copyfile(out_dir_temp + '\\' + name + '_T1_0000_reg.nii.gz', out_dir + '\\' + name + '_T1_reg.nii.gz' )
shutil.copyfile(out_dir_temp + '\\' + name + '_T1ce_0000_reg.nii.gz', out_dir + '\\' + name + '_T1ce_reg.nii.gz' )
shutil.copyfile(out_dir_temp + '\\' + name + '_T2_0000_reg.nii.gz', out_dir + '\\' + name + '_T2_reg.nii.gz' )
shutil.copyfile(out_dir_temp + '\\' + name + '_FLAIR_0000_reg.nii.gz', out_dir + '\\' + name + '_FLAIR_reg.nii.gz' )

shutil.copyfile(out_dir_temp + '\\' + name + '_Map5_0000_reg.nii.gz', out_dir + '\\' + name + '_Map5_reg.nii.gz' )
shutil.copyfile(out_dir_temp + '\\' + name + '_Map10_0000_reg.nii.gz', out_dir + '\\' + name + '_Map10_reg.nii.gz' )
shutil.copyfile(out_dir_temp + '\\' + name + '_Map15_0000_reg.nii.gz', out_dir + '\\' + name + '_Map15_reg.nii.gz' )

## removing all temp data or Continuously


## -----------  Merging into 4D nifti ----------  ##☺

D = [ f for f in os.listdir(out_dir_temp) if os.path.isfile(out_dir_temp+'\\'+f) if (out_dir_temp+'\\'+f).__contains__('.nii') ]

DTI = [ f for f in D if (out_dir_temp+'\\'+f).__contains__('DTI') ]
DCE = [ f for f in D if (out_dir_temp+'\\'+f).__contains__('DCE') ]
DSC = [ f for f in D if (out_dir_temp+'\\'+f).__contains__('DSC') ]
# map5 = [ f for f in D if (out_dir_temp+'\\'+f).__contains__('Map5') ]
# map10 = [ f for f in D if (out_dir_temp+'\\'+f).__contains__('Map10') ]
# map15 = [ f for f in D if (out_dir_temp+'\\'+f).__contains__('Map15') ]

path_save_comp = out_dir + '\\' + name + '_DTI.nii.gz'
util.merge_Nifti(DTI, out_dir_temp, path_save_comp)

path_save_comp = out_dir + '\\' + name + '_DCE.nii.gz'
util.merge_Nifti(DCE, out_dir_temp, path_save_comp)

path_save_comp = out_dir + '\\' + name + '_DSC.nii.gz'
util.merge_Nifti(DSC, out_dir_temp, path_save_comp)

# path_save_comp = out_dir + '\\' + name + '_Map5.nii.gz'
# util.merge_Nifti(map5, out_dir_temp, path_save_comp)

# path_save_comp = out_dir + '\\' + name + '_Map10.nii.gz'
# util.merge_Nifti(map10, out_dir_temp, path_save_comp)

# path_save_comp = out_dir + '\\' + name + '_Map15.nii.gz'
# util.merge_Nifti(map15, out_dir_temp, path_save_comp)

# remove remaining data separated ------------ ###


## -----------  Time-registration of 4D data ----------  ##




## -----------  Computation fo FA and ADC from DTI ----------  ##

file_path = data_directory + '\\DTI'
os.makedirs(out_dir_temp + '\\dti')
cmd = 'dcm2niix -b y -z y -x n -t n -m n -o ' + out_dir_temp + '\\dti' + ' -s n -v n ' + file_path
os.system(cmd)
data_fname = glob.glob(out_dir_temp + '\\dti\\*.nii.gz')[0]
bval_fname = glob.glob(out_dir_temp + '\\dti\\*.bval')[0]
bvec_fname = glob.glob(out_dir_temp + '\\dti\\*.bvec')[0]
data_fname = glob.glob(data_directory + '\\Outputs\\*DTI.nii.gz')[0]
mask_fname = glob.glob(data_directory + '\\Outputs\\*mask*.nii.gz')[0]


data, affine  = load_nifti(data_fname, return_img=False,return_coords=False)
mask,_  = load_nifti(mask_fname, return_img=False,return_coords=False)

bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
gtab = gradient_table(bvals, bvecs)


maskdata=np.zeros_like(data)
for i in range(data.shape[3]):
    maskdata[...,i] = data[...,i]*mask

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata)
tensor_vals = dti.lower_triangular(tenfit.quadratic_form)

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
save_nifti(data_directory + '\\Outputs\\' + 'FA_sitk.nii.gz', FA.astype(np.float32), affine)

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)
save_nifti(data_directory + '\\Outputs\\' + 'FA_RGB_sitk.nii.gz', np.array(255 * RGB, 'uint8'), affine)


MD = dti.mean_diffusivity(tenfit.evals)
MD[np.isnan(MD)] = 0
save_nifti(data_directory + '\\Outputs\\' + 'MD_sitk.nii.gz', MD.astype(np.float32), affine)

# MD1 = np.clip(MD, 0, 0.005)
# viewer = napari.view_image(np.ndarray.transpose(MD1,[2,1,0]))


# plt.figure
# np.histogram(MD1,256)


## -----------  Computation DCE analysis----------  ##

from fabber import Fabber
fab = Fabber()

fab.get_models()



from fabber import FabberCl
fab = FabberCl()

fab.get_models()


#%%
import os

cmd = 'C:\\Users\jakubicek\\envs\\MRI_Brain\\Scripts\\quantiphyse'
os.system(cmd)

import quantiphyse 








