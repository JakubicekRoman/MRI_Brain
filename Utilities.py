# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:14:02 2022

@author: jakubicek
"""


import numpy as np
import SimpleITK as sitk   
import matplotlib.pyplot as plt
from skimage import measure


def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image) 



def resave_dicom(data_directory, out_dir, name, ser, info, bias=True):
    
    ## for load dicom
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory+'\\')
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    
    series_file_names = series_file_names[info['vol']*ser :  info['vol']*(ser+1)]
    
    if info['Z-direction']:
        series_file_names = series_file_names[::-1]
        
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.LoadPrivateTagsOn()
    image3D=series_reader.Execute()
    sizeImg = image3D.GetSize()
    # spacing = image3D.GetSpacing()
    
    # orig = series_reader.Execute()
    # size = orig.GetSize()

    # resamplig and rotating
    image_out = []
    image_out = sitk.Image( sizeImg , sitk.sitkInt16)
    image_out = sitk.DICOMOrient(series_reader.Execute(), 'LPS')
    image_out = resample_image(image_out, out_spacing=[1.0, 1.0, 1.0], is_label=False)
    
    if bias:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # corrector.SetMaximumNumberOfIterations((100,100,100,100))
        corrector.SetMaximumNumberOfIterations((100,50))
        corrector.SetNumberOfHistogramBins(10)
        image_out = sitk.Cast(image_out, sitk.sitkFloat32)
        image_out = corrector.Execute(image_out)
        
    image_out = sitk.Cast(image_out, sitk.sitkInt16)
    
    # sitk.Show(orig, debugOn=True)
    # sitk.Show(image_out, debugOn=True)
    
    path_save_temp = out_dir + '\\' + name + '_' + str(ser) + '.nii.gz'
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(path_save_temp)
    writer.Execute(image_out)
    # sitk.WriteImage(image_out, path_save_temp)
    
    # return path_save_temp, info
    return path_save_temp


def read_nii(file_name, current_index):   
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("NiftiImageIO")
    file_reader.SetFileName(file_name)
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
            
    if current_index==-1:

        img = sitk.GetArrayFromImage(file_reader.Execute())
        # img = np.transpose(img, (1,2,0))
        
    else:
        if (current_index<1 and current_index>0):
            current_index = int( np.round( current_index*sizeA[2] ))
        current_index =  (0, 0, current_index, 0)
        file_reader.ReadImageInformation()
        file_reader.SetExtractIndex(current_index)
        extract_size = (sizeA[0], sizeA[1], 1, 1)
        file_reader.SetExtractSize(extract_size)

        img = sitk.GetArrayFromImage(file_reader.Execute())
        img = np.squeeze(img)
    
    
    # img = np.pad(img,((addX[2],addX[3]),(addX[0],addX[1]),(0,0)),'constant',constant_values=(-1024, -1024))


    return img


def read_nii_info(file_name):   
    
    info = dict()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    file_reader.ReadImageInformation()
    sizeA = file_reader.GetSize()
    
    info['size'] = (sizeA[0], sizeA[1])
    info['slices'] = (sizeA[2])
    info['Z-direction'] = False
    
    return info


def write_nii(A, file_name):  
    # A = np.transpose(A, (2,0,1))
    A = sitk.GetImageFromArray(A)
    sitk.Cast( A , sitk.sitkInt16)
    # A.SetOrigin((0, 0, 0))
    # A.SetSpacing((1,1,1))
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(file_name)
    writer.Execute(A)



def read_dicom_info(data_directory):   
    
    info = dict()
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory+'\\')
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    
    slices_loc = []
    for i,item in enumerate(series_file_names):
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(item)    
        file_reader.ReadImageInformation()
        slices_loc.append(file_reader.GetMetaData('0020|1041'))
    
    u_loc = unique(slices_loc)
    info['vol'] = len(u_loc)
    info['series'] =  len(slices_loc) / len(u_loc)
    info['Z-direction'] = (float(slices_loc[0])-float(slices_loc[1]))>0
    
    return info



def unique(list1):
  
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


def resize_with_padding(img, expected_size):
    delta_width = expected_size[0] - img.shape[0]
    delta_height = expected_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])
    padding[padding<0]=0
    img = np.pad(img, [(padding[0], padding[2]), (padding[1], padding[3])], mode='constant')
    img = crop_center(img, new_width=expected_size[0], new_height=expected_size[1])
    return img


def crop_center(img, new_width=None, new_height=None):        
    width = img.shape[1]
    height = img.shape[0]
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)  
        
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
        z = 1;
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
        z = img.shape[2]   
        
    return center_cropped_img




def display_reg(path_ref,path_mov,path_save_reg, sl):
    
    ##----- display results of reg ----- 
    # sl = 0.6
    img_ref = read_nii(path_ref, sl)
    img_mov = read_nii(path_mov, sl)
    img_reg = read_nii(path_save_reg, sl)
    
    img_ref = (( img_ref - img_ref.min() ) / (img_ref.max() - img_ref.min()) )
    img_mov = (( img_mov - img_mov.min() ) / (img_mov.max() - img_mov.min()) )
    img_reg = (( img_reg - img_reg.min() ) / (img_reg.max() - img_reg.min()) )
    
    
    img_mov = resize_with_padding(img_mov, img_ref.shape )
    
    # fig, axs = plt.subplots(2,2)
    # axs[0,0].imshow(img_ref, cmap='gray')
    # axs[0,1].imshow(img_mov, cmap='gray')
    # axs[1,0].imshow(img_ref, cmap='gray')
    # axs[1,1].imshow(img_reg, cmap='gray')
    
    img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
    img[:,:,0] = img_mov
    img[:,:,1] = img_ref
    img[:,:,2] = img_mov
    
    # plt.figure()
    # plt.imshow(img)
    
    fig, axs = plt.subplots(1,2)        
    axs[0].imshow(img)
    
    img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
    img[:,:,0] = img_reg
    img[:,:,1] = img_ref
    img[:,:,2] = img_reg
    
   
    # plt.figure()
    # plt.imshow(img)

    axs[1].imshow(img)
    plt.show()

   
    
def bound_3D(img, add):    
    # mask.ndim
    # g = np.where(mask.sum(axis=0)>0)
    loc = np.where(img>0)

    borders=[]
    for i in range(0,len(loc)):
        g = [int(loc[i].min()), int(loc[i].max())]
        borders.append(g[:])
    
    return borders


def bwareafilt(mask, n=1, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    # For openCV > 3.0 this can be changed to: areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels = measure.label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas





