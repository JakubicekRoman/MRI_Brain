import os
import numpy as np
# import matplotlib.pyplot as plt
# import time
import nibabel as nib




def check_orientation(ct_arr, coord):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    (x,y,z) = coord

    if x != 'R':
        # ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
        ct_arr = np.flip(ct_arr, axis=0)
    if y != 'P':
        # ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
        ct_arr = np.flip(ct_arr, axis=1)
    if z != 'S':
        # ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
        ct_arr = np.flip(ct_arr, axis=2)
    return ct_arr