# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:28:43 2022

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


###### rename files for onetimes

# pathDir = 'C:\Data\Jakubicek\MRI_Brain\Ambrozek\Outputs'
# pathDir = 'C:\Data\Jakubicek\MRI_Brain\Bednarova\Outputs'
pathDir = 'C:\Data\Jakubicek\MRI_Brain\Cip\Outputs'

D = [ f for f in os.listdir(pathDir) if os.path.isfile(pathDir+'\\'+f) if (pathDir+'\\'+f).__contains__('.nii') if not (pathDir+'\\'+f).__contains__('mask') ]


for f in D:
    name_old = f.split('_')[2]
    name_new = '00' + name_old
    name_new = name_new[-3:]
    os.rename( pathDir+'\\'+f, pathDir+'\\'+f.replace(name_old,name_new) )
    

######