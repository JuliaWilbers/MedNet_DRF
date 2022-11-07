'''Dataset for training'''

import math
import os
import random
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import torch
import cv2

class DRF_data(Dataset):

    def __init__(self, data_dir, label_list, sets):
        with open(label_list, 'r') as f:
            self.label_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.label_list)))
        
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.set_size = len(self.label_list)
        self.data_dir = data_dir

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data
   
    
    def __len__(self):
        return self.set_size
    
    
    def __getitem__(self, idx):
    
        # Read image and segmentation
        ith_info = self.label_list[idx].split(" ")
        self.name = ith_info[0] 
        self.img_name = os.path.normpath(os.path.join(self.data_dir, self.name, "image.nii.gz"))
        self.seg_name = os.path.normpath(os.path.join(self.data_dir, self.name, "mask.nii.gz"))
            
        # Read labels 
        label = torch.tensor([float(ith_info[1])])
        assert os.path.isfile(self.img_name)
            
        img = nibabel.load(self.img_name)  
        mask = nibabel.load(self.seg_name)
        assert img is not None
        assert mask is not None
        
        if self.phase == "train":
            
            # data processing
            img_array = self.__training_data_process__(img)
            mask_array = self.__training_data_process__(mask)
                
            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)
            
            return img_array, mask_array, label
            
        elif self.phase == "test":

            # data processing
            img_array = self.__testing_data_process__(img)
            mask_array = self.__testing_data_process__(mask)
            
            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            return img_array, masks, label, self.name

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        """
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        """
        """
        pixels = volume
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        
        """
        pixels = volume
        _max = pixels.max()
        _min = pixels.min()
        
        out = (pixels - _min)/(_max-_min)
        
        return out

    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()
        #print("before normalisation, ", "max: ", data.max(), "min: ", data.min())
        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
        #print("after normalisation, ", "max: ", data.max(), "min: ", data.min())
        return data

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data