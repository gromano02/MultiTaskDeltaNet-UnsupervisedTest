import os
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import CDDataAugmentation
#from data_utils import CDDataAugmentation


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
                with_random_rot = True,        
                with_scale_random_crop=True,  #### False
                with_random_blur=True,        #### False 
                random_color_tf=True          #### False
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        #name = self.img_name_list[index]
        #A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        #B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(self.ref)
        img_B = np.asarray(self.res)

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size

class REGDataset(ImageDataset):

    def __init__(self, ref, res, label1, label2, ref_y,res_y, max_, min_, Len, Pi, img_size, is_train=True,
                 to_tensor=True):
        super(REGDataset, self).__init__(ref, res, img_size=img_size, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label1 = label1
        self.label2 = label2
        self.ref_y  = ref_y
        self.res_y  = res_y
        #self.ref_t  = ref_t
        #self.res_t  = res_t
        self.max_  = max_
        self.min_  = min_
        self.Len    = Len
        self.Pi     = Pi

    def __getitem__(self, index):

        img = np.asarray(self.ref[index % self.A_size])
        img_B = np.asarray(self.res[index % self.A_size])

        label1 = np.array(self.label1[index % self.A_size], dtype=np.uint8)
        label2 = np.array(self.label2[index % self.A_size], dtype=np.uint8)

        #y      = self.y[index % self.A_size]    
        ref_y   = self.ref_y[index % self.A_size]
        res_y   = self.res_y[index % self.A_size]
        #ref_t   = self.ref_t[index % self.A_size]
        #res_t   = self.res_t[index % self.A_size]
        max_   = self.max_[index % self.A_size]
        min_   = self.min_[index % self.A_size]
        Len    = self.Len[index % self.A_size]
        Pi     = self.Pi[index % self.A_size]

        # if you are getting error because of dim mismatch ad [:,:,0] at the end

        [img, img_B], [label1,label2] = self.augm.transform([img, img_B], [label1,label2], to_tensor=self.to_tensor)
        # print(label.max())

        return {'A': img, 'B': img_B, 'L1': label1, 'L2': label2, 'y1':ref_y, 'y2':res_y, 'max':max_, 'min':min_, 'Len':Len, 'Pi':Pi}


