import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch



class StyleTransferDataset(Dataset):
    """Coco Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.names = os.listdir(root_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.names[idx])
        
        #print (image_name)
        image = mpimg.imread(image_name)
        
        #image = Image.open(image_name)
        #print (type (image))
        #print (image.getdata(image))
        #plt.imshow(image)
        
        #plt.show()
        
        
        #some images in coco data set were gray. Since there is nothing that prohibits from doing style transfer, I simply convert it to rgb
        if len(image.shape) == 2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        # if image has an alpha color channel, get rid of it
        elif(image.shape[2] == 4):
            image = image[:,:,0:3]
            
        
 
        pil_image = Image.fromarray(image)
        if self.transform:
            sample = self.transform(pil_image)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image =sample['image']
        
        image_copy = np.copy(image)
        
        # convert image to grayscale
        
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}