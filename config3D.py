import numpy as np
import torch

# from utils import create_masks_hamming
# from utils import create_masks_crop

class Config:
    
    # data_path = r'C:\Data\Jakubicek\MRI_Brain\Ambrozek\Outputs\comp'
    # data_path = r'C:\Data\Jakubicek\MRI_Brain\Bednarova\Outputs\comp'
    data_path = r'C:\Data\Jakubicek\MRI_Brain\Cip\Outputs\comp'

    seq = 'DCE_'
    # seq = 'DSC_'
    # seq = 'DTI_'
    
    save_sufix = seq + 'reg3'
    # save_sufixO = seq + 'orig3'
    
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    
    
    np_dtype = np.float32
    torch_dtype = torch.float32

    # scales = np.array([np.sqrt(128),np.sqrt(64),np.sqrt(32), np.sqrt(16), np.sqrt(8), np.sqrt(4), np.sqrt(2), 1])
    scales = np.array([1])
    sigmas = scales - 1
    
    # init_lr = 0.00001
    # init_lr = 0.0001  #fungujepro masku
    init_lr = 0.002  #fungujepro masku
    # init_lr = 0.001

    resize_factor = 1
    
    
    # iterations = [80,100,120]
    # iterations = [300,380,400]
    iterations = [80,120,150]
    
    gamma = 0.1
    
    num_batches = 1
    
    interp_mode = 'bilinear'
    # interp_mode = 'bicubic' 
    # interp_mode = 'nearest'
    
    align_corners = False


    # create_masks = create_masks_hamming(crop_fraction=0.1, np_dtype=np_dtype)
    # create_masks = create_masks_crop(crop_fraction=0.1, np_dtype=np_dtype) 
    
    pad = 0
    
    regularization_factor = 1e-2
    # regularization_factor2 = 0.5