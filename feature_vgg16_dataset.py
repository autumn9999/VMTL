
#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path
import scipy.io as ioc
import pdb



def make_dataset_test(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def pil_loader(path, dataset):

    # for office-home
    if dataset == "office-home":
        path = '/Dataset/'+path.split('data/')[-1]
    img = Image.open(path)
    return img.convert('RGB'), path    

class ImageList_feature(object):
    def __init__(self, image_list, dataset, labels=None, transform=None):
        
        self.imgs = make_dataset_test(image_list, labels)
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.transform = transform
        self.loader = pil_loader
        self.dataset = dataset

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img, new_path = self.loader(path, self.dataset)
        if self.transform is not None:
            img = self.transform(img)
        return img, new_path

    def __len__(self):
        return len(self.imgs)



