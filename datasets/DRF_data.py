'''
Dataset for training'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import torch


class DRF_data(Dataset):

    def __init__(self, im_dir, seg_dir, img_list, label_list, data_size, sets):
        with open(label_list, 'r') as f:
            self.label_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.label_list)))
        
        #self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.set_size = data_size
        self.im_dir = im_dir
        self.seg_dir = seg_dir
        
        """
        # for testing
        idx = 1
        ith_info = self.label_list[idx].split(" ")
        self.name = ith_info[0] + "nii.gz"
        self.img_name = os.path.normpath(os.path.join(self.im_dir, self.name))
        self.label_name = os.path.normpath(os.path.join(self.seg_dir, self.name))
        print(self.img_name)
        """

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data
    """
    def __len__(self):
        return len(self.img_list)
        
    """
    
    def __len__(self):
        return self.set_size
    
    
    def __getitem__(self, idx):

        if self.phase == "train":
            # read image and labels
            ith_info = self.label_list[idx].split(" ")
            self.name = ith_info[0] + ".nii.gz"
            self.img_name = os.path.normpath(os.path.join(self.im_dir, self.name))
            #self.label_name = os.path.normpath(os.path.join(self.seg_dir, self.name))
            #label = torch.tensor([float((ith_info[1])), float((ith_info[2]))])
            label = torch.tensor([float(ith_info[1])])
            assert os.path.isfile(self.img_name)
            #assert os.path.isfile(self.label_name)
            img = nibabel.load(self.img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            #mask = nibabel.load(self.label_name)
            #assert mask is not None
            #
            #label = torch.Tensor([label]) 
            
            # data processing
            img_array = self.__training_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            #mask_array = self.__nii2tensorarray__(mask_array)

            #assert img_array.shape == mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(
            #img_array.shape, mask_array.shape)
            #return img_array, mask_array, label
            return img_array, label

        elif self.phase == "test":
            # read image
            #ith_info = self.img_list[idx].split(" ")
            #img_name = os.path.join(self.root_dir, ith_info[0])
            img_name = im_dir
            print(img_name)
            assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            assert img is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            return img_array

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __testing_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data