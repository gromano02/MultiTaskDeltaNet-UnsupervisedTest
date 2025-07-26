"""
Vertical line dataset
"""

import os
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import CDDataAugmentation

class ImageDataset(data.Dataset):
    """Material dataloder"""
    def __init__(self, ref, res, img_size=128, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()

        self.ref = ref
        self.res = res
        #self.label = label
        self.img_size = img_size
        self.A_size = len(self.ref)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_random_rot = True,       #### False,
                with_scale_random_crop=True,  #### False
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):

        img = np.asarray(self.ref)
        img_B = np.asarray(self.res)

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class VERDataset(ImageDataset):

    def __init__(self, ref, res, label, img_size, is_train=True,
                 to_tensor=True):
        super(VERDataset, self).__init__(ref, res, img_size=img_size, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label = label

    def __getitem__(self, index):

        img = np.asarray(self.ref[index % self.A_size])
        img_B = np.asarray(self.res[index % self.A_size])

        label = np.array(self.label[index % self.A_size], dtype=np.uint8)
        # if you are getting error because of dim mismatch ad [:,:,0] at the end

        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'L': label}


class CARDataset(ImageDataset):

    def __init__(self, ref, res, label1, label2, img_size, is_train=True,
                 to_tensor=True):
        super(CARDataset, self).__init__(ref, res, img_size=img_size, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label1 = label1
        self.label2 = label2

    def __getitem__(self, index):

        img = np.asarray(self.ref[index % self.A_size])
        img_B = np.asarray(self.res[index % self.A_size])

        label1 = np.array(self.label1[index % self.A_size], dtype=np.uint8)
        label2 = np.array(self.label2[index % self.A_size], dtype=np.uint8) 
        # if you are getting error because of dim mismatch ad [:,:,0] at the end

        [img, img_B], [label1,label2] = self.augm.transform([img, img_B], [label1,label2], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'L1': label1, 'L2': label2}
