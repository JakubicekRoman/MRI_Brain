# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:03:55 2022

@author: jakubicek
"""
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import nibabel as nib
from scipy import ndimage
import Utilities as util

from config2D import Config
from Utilities import construct_transf_matrix
from losses import weighted_MSE_valid
from losses import diagonal_regularization


config = Config()
torch.set_default_dtype(config.torch_dtype)

# file_names = glob(config.data_path + '/**/*.avi',recursive=True)
# file_names = [f for f in file_names if not 'registered' in f]

file_names = glob(config.data_path + os.sep + config.seq + '.nii.gz',recursive=True)



for file_num, file_name in enumerate(file_names):
    
    img = nib.load(file_name)
    imgs = img.get_fdata()
    
    imgs = imgs.astype(config.np_dtype)
    
    # imgs = imgs[:,:,30,1:10:1]
    imgs = imgs[:,:,30,1:]
    
    # imgs[:,:,5] = ndimage.rotate(imgs[:,:,5],10, reshape=False)
    # util.show_3D(imgs)
    
    imgs = np.transpose(imgs,[2,0,1])
    
    # imgs = load_video(file_name,config.np_dtype)
    
    # masks = config.create_masks(imgs.shape)
    
    # imgs,masks = padding(imgs,masks,config.pad)

    angle_rad = torch.zeros(imgs.shape[0]).to(config.device)
    tx = torch.zeros(imgs.shape[0]).to(config.device)
    ty = torch.zeros(imgs.shape[0]).to(config.device)
    angle_rad.requires_grad = True
    tx.requires_grad = True
    ty.requires_grad = True
    
    params = [angle_rad,tx,ty]


    losses = []
    for scale_num, (scale, sigma) in enumerate(zip(config.scales,config.sigmas)):
        
    #     imgs_res, masks_res = resize_pyramid(imgs, masks, scale, sigma, config.np_dtype)
        imgs_res = imgs.copy()
        
        imgs_res = torch.unsqueeze(torch.from_numpy(imgs_res),1)
    #     masks_res = torch.unsqueeze(torch.from_numpy(masks_res),1)

        optimizer = torch.optim.Adam(params,lr=config.init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.iterations, gamma=config.gamma, last_epoch=-1)

        
        for it in range(config.iterations[-1]):

            output = []
            loss_tmp = []
            randomizer = util.Randomizer(config.num_batches, imgs_res.shape[0])
            # randomizer = [0,1,2]
            for batch_num, inds in enumerate(randomizer):
                
                imgs_res_batch = imgs_res[inds,:,:,:].to(config.device)
    #             masks_res_batch = masks_res[inds,:,:,:].to(config.device)
                
                theta = construct_transf_matrix(params, config.resize_factor, inds, config.device)
                
                
                grid = F.affine_grid(theta[:,0:2,:], imgs_res_batch.shape, align_corners=config.align_corners)
                
                output_batch = F.grid_sample(imgs_res_batch, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
                # with torch.no_grad(): 
    #                 out_masks = F.grid_sample(masks_res_batch, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
                out_masks = torch.ones(output_batch.shape).to(torch.float32).to(config.device)
                
                loss1 = weighted_MSE_valid(output_batch,out_masks)
    #             loss2 = diagonal_regularization(config.resize_factor, theta, config.device)
                
    #             loss = loss1 + config.regularization_factor * loss2
                loss = loss1 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                output.append(output_batch[:,0,:,:].detach().cpu().numpy())
                
                loss_tmp.append(loss.detach().cpu().numpy())
                
                
                
            output = np.concatenate(output,0)
            output= randomizer.order_corection(output)
            losses.append(np.mean(loss_tmp))
            
            if (it % 100) == 0:
                print(it)
                print(theta[0,:,:])
                # print(loss1,loss2)
                print(loss1)
                
                plt.plot(losses)
                plt.show()
                
                std1 = np.std(imgs_res[:,0,:,:].detach().cpu().numpy(), axis=0)
                std2 = np.std(output, axis=0)   
                plt.imshow(np.concatenate( (std1, std2), axis=1))
                plt.show()
                
                # plt.imshow(imgs_res[0,0,:,:].detach().cpu().numpy()-imgs_res[1,0,:,:].detach().cpu().numpy())
                # plt.imshow(output[0,:,:] - output[8,:,:])
                plt.imshow(imgs_res[1,0,:,:].detach().cpu().numpy() - output[5,:,:])
                # plt.imshow(std1 - std2)
                plt.show()
    
    
    res = np.transpose(output,[1,2,0])
    res = res.astype(np.int16)
    res = nib.Nifti1Image(res, np.eye(4))
    res.get_data_dtype() == np.dtype(np.int16)
    nib.save(res, os.path.join(config.data_path, config.save_sufix+'.nii.gz'))  

    orig = np.transpose(imgs,[1,2,0])
    orig = orig.astype(np.int16)
    orig = nib.Nifti1Image(orig, np.eye(4))
    orig.get_data_dtype() == np.dtype(np.int16)
    nib.save(orig, os.path.join(config.data_path, config.seq+'orig.nii.gz'))  