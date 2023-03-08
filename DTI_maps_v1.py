# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:55:08 2023

@author: jakubicek
"""

# import sys
# import numpy as np
# import numpy.matlib
import os
import shutil
# import random
# import pydicom as dcm
# import SimpleITK as sitk   
# import matplotlib.pyplot as plt
# import copy
# import nibabel as nib

# import Utilities as util

# import nipype.interfaces.dcm2nii as Dcm2nii
# import nipype.interfaces.dcm2nii
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces import niftyfit


pat = 'C:\Data\MRI_Glioms\MRI_Brain\Bogner'

data_directory = (pat)

name = os.path.basename(data_directory)

out_dir =  pat + '\\Outputs_maps\\DTI'

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

# out_dir_temp = out_dir + '\\temp\\DTI'

# if os.path.exists(out_dir_temp):
#     shutil.rmtree(out_dir_temp)
# os.makedirs(out_dir_temp)


## resaving
converter = Dcm2niix()
converter.inputs.source_dir = pat + '\DTI'
# converter.inputs.gzip_output = True
converter.inputs.output_dir = out_dir
# converter.cmdline

# cmd = 'C:\\Users\jakubicek\anaconda3\envs\MRI_Brain\Lib\site-packages\nipype\interfaces\\' + converter.cmdline
cmd = converter.cmdline
converter.cmdline
os.system(cmd)

# # converter.run() 

#%%

dwi_tool = niftyfit.DwiTool(dti_flag=True)
dwi_tool.inputs.source_file = out_dir + os.sep +'DTI_GRANT_GLIOMY_V6_20220922130737_6.nii.gz'
dwi_tool.inputs.bvec_file = out_dir + os.sep +'DTI_GRANT_GLIOMY_V6_20220922130737_6.bvec'
dwi_tool.inputs.bval_file = out_dir + os.sep +'DTI_GRANT_GLIOMY_V6_20220922130737_6.bval'
# dwi_tool.inputs.mask_file = 'mask.nii.gz'
# dwi_tool.inputs.b0_file = 'b0.nii.gz'
# dwi_tool.inputs.rgbmap_file = 'rgb_map.nii.gz'
# dwi_tool. = out_dir
cmd = dwi_tool.cmdline

# cmd = cmd.replace('dwi_tool', 'dwi')
# cmd = 'C:\\Users\\jakubicek\\anaconda3\\envs\\MRI_Brain\\Lib\\site-packages\\nipype\\interfaces\\niftyfit\\' + cmd
os.system(cmd)

# dt = niftyfit.DwiTool()
# dt.inputs.source_file = out_dir + os.sep +'DTI_GRANT_GLIOMY_V6_20220922130737_6.nii.gz'
# dt.inputs.bval_file = out_dir + os.sep +'DTI_GRANT_GLIOMY_V6_20220922130737_6.bval'
# cmd = dt.cmdline
# dt.cmdline
# os.system(cmd)
