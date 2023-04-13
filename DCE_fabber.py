#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:56:07 2023

@author: jakubicek
"""

import os


path_data = '/home/jakubicek/Glioms_Brain/Outputs/Bogner_DCE.nii.gz'
path_save = '/home/jakubicek/Glioms_Brain/Outputs/Bogner_TOFTS'
path_mask = '/home/jakubicek/Glioms_Brain/Outputs/Bogner_mask_Brain.nii.gz'


model = 'dce_tofts'


cmd = 'fabber_dce --data=' + path_data + ' --mask=' + path_mask + ' --output=' + path_save
cmd = cmd + ' --model=' + model
cmd = cmd + ' --method=vb --noise=white --delt=0.1 --fa=15 --tr=0.0027 --r1=3.7 --delay=0.5 --aif=orton --infer-delay --infer-sig0 --infer-t10 --convergence=trialmode --max-trials=20' 
cmd = cmd + ' --overwrite'


os.system(cmd)

