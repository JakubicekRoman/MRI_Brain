# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:20:10 2022

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

from config3D import Config
from Utilities import construct_transf_matrix_3D
from losses import weighted_MSE_valid
from losses import diagW_regularization
from losses import transl_regularization


config = Config()
torch.set_default_dtype(config.torch_dtype)

# file_names = glob(config.data_path + '/**/*.avi',recursive=True)
# file_names = [f for f in file_names if not 'registered' in f]

file_names = glob(config.data_path + os.sep + config.seq + '.nii.gz',recursive=True)



for file_num, file_name in enumerate(file_names):
    
    img = nib.load(file_name)
    imgs = img.get_fdata()
    
    imgs = imgs.astype(config.np_dtype)
    
    # imgs = np.transpose(imgs,[2,1,0,3])
    
    imgs = imgs[:,:,::2,1::2]
    # imgs = imgs[:,:,30,1:]
    
    # imgs[:,:,5] = ndimage.rotate(imgs[:,:,5],10, reshape=False)
    # util.show_3D(imgs)
    
    imgs = np.transpose(imgs,[3,2,0,1])
    
    # imgs = load_video(file_name,config.np_dtype)
    
    # masks = config.create_masks(imgs.shape)
    
    # imgs,masks = padding(imgs,masks,config.pad)

    angle_rad_X = torch.zeros(imgs.shape[0]).to(config.device)
    angle_rad_Y = torch.zeros(imgs.shape[0]).to(config.device)
    angle_rad_Z = torch.zeros(imgs.shape[0]).to(config.device)
    tx = torch.zeros(imgs.shape[0]).to(config.device)
    ty = torch.zeros(imgs.shape[0]).to(config.device)
    tz = torch.zeros(imgs.shape[0]).to(config.device)
    
    angle_rad_X.requires_grad = True
    angle_rad_Y.requires_grad = True
    angle_rad_Z.requires_grad = True
    tx.requires_grad = True
    ty.requires_grad = True
    tz.requires_grad = True
    
    params = [angle_rad_X,angle_rad_Y,angle_rad_Z,tx,ty,tz]


    losses = []
    for scale_num, (scale, sigma) in enumerate(zip(config.scales,config.sigmas)):
        
    #     imgs_res, masks_res = resize_pyramid(imgs, masks, scale, sigma, config.np_dtype)
        # imgs_res = imgs.copy()
        imgs_res = imgs
        # imgs_res = imgs_res[5:-1,:,:,:]
        
        imgs_res = torch.unsqueeze(torch.from_numpy(imgs_res),1)
    #     masks_res = torch.unsqueeze(torch.from_numpy(masks_res),1)

        optimizer = torch.optim.Adam(params,lr=config.init_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.iterations, gamma=config.gamma, last_epoch=-1)

        m, s = torch.mean(imgs_res,(2,3,4)), torch.std(imgs_res,(2,3,4))
        imgs_res = (imgs_res - m[:,:,None,None, None]) / s[:,:,None,None,None]
        
        with util.Timer():
            for it in range(config.iterations[-1]):
    
                output = []
                loss_tmp = []
                randomizer = util.Randomizer(config.num_batches, imgs_res.shape[0])
                # randomizer = [0,1,2]
                for batch_num, inds in enumerate(randomizer):
                    
                    imgs_res_batch = imgs_res[inds,:,:,:].to(config.device)
        #             masks_res_batch = masks_res[inds,:,:,:].to(config.device)
                    
                    theta = construct_transf_matrix_3D(params, config.resize_factor, inds, config.device)
                    
                    
                    grid = F.affine_grid(theta[:,0:3,:], imgs_res_batch.shape, align_corners=config.align_corners)
                    
                    # m, s = torch.mean(imgs_res_batch,(2,3,4)), torch.std(imgs_res_batch,(2,3,4))
                    # imgs_res_batch = (imgs_res_batch - m[:,:,None,None, None]) / s[:,:,None,None,None]
                    
                    output_batch = F.grid_sample(imgs_res_batch, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
                    # with torch.no_grad(): 
        #                 out_masks = F.grid_sample(masks_res_batch, grid, padding_mode="zeros", align_corners=config.align_corners, mode=config.interp_mode)
                    out_masks = torch.ones(output_batch.shape).to(torch.float32).to(config.device)
                    W = np.zeros((4,4))
                    W[3,3] = 1
                    
                    loss1 = weighted_MSE_valid(output_batch,out_masks)
                    # loss2 = diagW_regularization(config.resize_factor, theta, W, config.device)
                    # loss2 = transl_regularization(theta, config.device)
                    
                    # loss = loss1 + config.regularization_factor * loss2
                    loss = loss1 
                    
                    # if it>0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    output.append(output_batch[:,0,:,:].detach().cpu().numpy())
                    
                    loss_tmp.append(loss.detach().cpu().numpy())
                    
                    
                    
                output = np.concatenate(output,0)
                output= randomizer.order_corection(output)
                losses.append(np.mean(loss_tmp))
                
                if (it % 2) == 1:
                    print(it)
                    print(theta[0,:,:])
                    # print(loss1,loss2)
                    print(loss1)
                    
                    plt.plot(losses)
                    plt.show()
                    
                    # imgs_res = imgs_res[:,0,:,:,:].detach().cpu().numpy()
                    # imgs_res = np.transpose(imgs_res.detach().cpu().numpy(),[0,1,2,3,4])
                    
                    sl_num = int(np.floor( imgs_res.shape[2]/2 ))
                    std1 = np.std(imgs_res[:,0,:,:,:].detach().cpu().numpy(), axis=0)
                    std2 = np.std(output, axis=0)   
                    std1 = np.max( std1, axis= 0)
                    std2 = np.max( std2, axis= 0)
                    plt.imshow(np.concatenate( (std1, std2), axis=1))
                    plt.show()
                    
                    # max1 = np.max(imgs_res[:,0,sl_num,:,:].detach().cpu().numpy(), axis=0)
                    # max2 = np.max(output[:,sl_num,:], axis=0)
                    # plt.imshow(np.concatenate( (max1, max2), axis=1))
                    # plt.show()
                    
                    # # plt.imshow(imgs_res[0,0,:,:].detach().cpu().numpy()-imgs_res[1,0,:,:].detach().cpu().numpy())
                    # # plt.imshow(output[0,:,:] - output[8,:,:])
                    # plt.imshow(imgs_res[1,0,sl_num,:,:].detach().cpu().numpy() - output[30,sl_num,:,:])
                    plt.imshow(np.abs(std1 - std2))
                    plt.show()
                
                
    
    res = ((output*s[:,None, None].numpy())+m[:,None, None].numpy()).astype(np.int16)
    res = np.transpose(res,[2,3,1,0])
    res = nib.Nifti1Image(res, np.eye(4))
    res.get_data_dtype() == np.dtype(np.int16)
    nib.save(res, os.path.join(config.data_path, config.save_sufix+'.nii.gz'))  

    # orig = np.transpose(imgs_res[:,0,:,:,:].detach().cpu().numpy(),[2,3,1,0])
    # orig = orig.astype(np.int16)
    # orig = nib.Nifti1Image(orig, np.eye(4))
    # orig.get_data_dtype() == np.dtype(np.int16)
    # nib.save(orig, os.path.join(config.data_path, config.save_sufixO+'_o.nii.gz'))  