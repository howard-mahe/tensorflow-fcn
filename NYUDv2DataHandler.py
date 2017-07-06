import numpy as np
import tensorflow as tf

import random
from PIL import Image

# Inspired by: https://github.com/shoaibahmed/FCN-TensorFlow/blob/master/inputReader.py

NYUDv2_BGR_MEAN = (116.190, 97.203, 92.318)
NYUDv2_LOGD_MEAN = (7.844,)

class NYUDv2DataHandler:
    def __init__(self, data_type, num_classes):
        # arguments
        self.num_classes = num_classes
        self.data_type   = data_type

        # Parameters
        self.H        = 425
        self.W        = 560
        self.nyud_dir = '/home/howard/dataset/nyud-rgbd/data/'
        self.seed     = 1337

        # Set # of channels
        if self.data_type == 'BGR':
            self.channels = 3
        elif self.data_type == 'BGR-D':
            self.channels = 4
        else:
            raise ValueError('input_type not supported')

        # Mean
        self.mean_bgr  = np.array(NYUDv2_BGR_MEAN, dtype=np.float32)
        self.mean_logd = np.array(NYUDv2_LOGD_MEAN, dtype=np.float32)  
        
        # load indices for images and labels
        split_f = '{}/{}.txt'.format(self.nyud_dir, 'trainval')
        self.train_indices = open(split_f, 'r').read().splitlines()
        split_f = '{}/{}.txt'.format(self.nyud_dir, 'test')
        self.test_indices = open(split_f, 'r').read().splitlines()
        self.last_test_idx = 0

        # randomization
        random.seed(self.seed)
        
    def create_placeholders(self):
        self.input_image = tf.placeholder(dtype=tf.float32,
            shape=[None, self.H, self.W, self.channels], name="inputImage")

        self.label = tf.placeholder(dtype=tf.uint8,
            shape=[None, self.H, self.W, 1], name="inputLabel")

        return self.input_image, self.label
   
    def get_sample(self, phase, batch_size):
        # trainval samples are shuffled
        if phase == 'train':
            indices = self.train_indices
            self.idx = np.random.randint(0, len(indices)-1, size=batch_size)
            self.last_test_idx = 0
        # test samples are picked sequentially
        elif phase == 'test':
            indices = self.test_indices            
            self.idx = [x%len(indices) for x in np.arange(self.last_test_idx, self.last_test_idx + batch_size)]
            self.last_test_idx += batch_size
        
        # load input data
        for k in range(batch_size):
            image = self.load_image_from_disk(indices[self.idx[k]])
            label = self.load_label_from_disk(indices[self.idx[k]])
            if k==0:
                images = np.zeros((batch_size,)+image.shape)
                labels = np.zeros((batch_size,)+label.shape)
            images[k,:,:,:] = image
            labels[k,:,:,:] = label

        return images, labels

    def load_image_from_disk(self, idx):
        im = Image.open('{}/images/img_{}.png'.format(self.nyud_dir, idx))
        rgb = np.array(im, dtype=np.float32)
        bgr = rgb[:,:,::-1]  # conversion to BGR
        bgr -= self.mean_bgr # mean substraction

        if self.data_type == 'BGR':
            return bgr

        elif self.data_type == 'BGR-D':
            im = Image.open('{}/depth/img_{}.png'.format(self.nyud_dir, idx))
            d = np.array(im, dtype=np.float32)
            d = np.log(d)
            d -= self.mean_logd     # mean substraction
            #d /= self.D_DIVISOR    # depth normalization
            d = d[..., np.newaxis]
            return np.dstack((bgr, d)) # stack bgr & d images

    def load_label_from_disk(self, idx):
        """
        Consumes an image index and returns corresponding image
		Args:
		  idx: Image index
		Returns:
		  One 3-D numpy arrays: The label image
		"""
        im = Image.open('{}/label40/img_{}.png'.format(self.nyud_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[..., np.newaxis]
        return label
