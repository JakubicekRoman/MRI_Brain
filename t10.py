#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:59:35 2023

@author: jakubicek
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
# sys.path.append('../src')
import t1_fit
import napari


# # Fit data:
# s = np.array([413, 604, 445])
# tr = 5.4e-3
# fa = np.array([2, 5, 12])

# s0, t1 = t1_fit.VFA2Points(fa, tr).proc(s)

# fa_range = np.linspace(0, 20, 50)

# # Plot data:
# print(f"Fitted values: s0 = {s0:.1f}, t1 = {t1:.3f} s")
# plt.plot(fa_range, t1_fit.spgr_signal(s0=s0,  t1=t1, tr=tr, fa=fa_range), '-', label='model')
# plt.plot(fa, s, 'o', label='signal')
# plt.xlabel('FA (deg)')
# plt.ylabel('signal');
# plt.legend();

path_data = '/home/jakubicek/Glioms_Brain/Outputs'
path_save = '/home/jakubicek/Glioms_Brain/Outputs/Bogner_TOFTS'

vfa_fitter = t1_fit.VFALinear(fa=[5, 10, 15], tr=0.0027)

maps = []
for file_name in os.listdir(path_data):
    if 'Map' in file_name:
        # print(file_name)
        maps.append(os.path.join( path_data, file_name))

s0, t1 = vfa_fitter.proc_image(maps, threshold=50, dir=path_save, suffix='_VFA',n_procs=10);



