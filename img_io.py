"""
 " Description: TensorFlow ExpandNet for HDR image reconstruction.
 " Author: LiJinghui
 " Date: March 2018
"""
from __future__ import division
import numpy as np
import cv2
import os, random, glob
import tensorflow as tf

patch_size = 256

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def read_LDR(sdr_img):
    image_sdr_raw = cv2.imread(sdr_img)
    if image_sdr_raw is None:
        print('Could not load {0}'.format(sdr_img))
    image_sdr_input = map_range(np.float32(image_sdr_raw))
    image_sdr_input = replace_specials_(image_sdr_input)
    image_sdr_input = image_sdr_input[np.newaxis, :, :, :]
    return image_sdr_raw, image_sdr_input

# Write LDR/HDR image
def write_LDR(img, file_path):
    img = img.astype(np.uint8)
    cv2.imwrite(file_path,img)

def write_HDR(img, file_path):
    img = img * 65535
    img = img.astype(np.uint16)
    cv2.imwrite(file_path,img)


def data_aug_tf(image, label):
    # data_aug_flip_lr
    index = random.randint(1,2)
    if index == 1:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    # data_aug_flip_ud
    index = random.randint(1,2)
    if index == 1:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)

    # data_aug_rot
    index = random.randint(0,4)
    image = tf.image.rot90(image,index)
    label = tf.image.rot90(label,index)

    return image,label

def data_aug(image, label):
    flag = random.randint(1, 2)
    if flag == 1:
        index = random.randint(-1,1)
        image = cv2.flip(image, index)
        label = cv2.flip(label, index)
    else :
        image, label = image, label
    return image, label

def add_noise(x):
    return x + 0.1 * x.std()*np.random.random(x.shape)

def normalize_imgs_fn(x, type):
    if type == 'sdr':
        x = x / 255.
    else:
        x = np.divide(x, 65535.)
    return x

def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

def replace_specials_(x, val=0):
    x[np.isinf(x).sum() | np.isnan(x).sum()] = val
    return x

# Read training data (HDR ground truth and LDR JPEG images)
def load_training_pair(sdr_img, hdr_img):
    # Read JPEG LDR image and HDR ground truth
    input_image_sdr = cv2.imread(sdr_img)
    input_image_hdr = cv2.imread(hdr_img, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
    # random flip image and label
    input_image_sdr, input_image_hdr = data_aug(input_image_sdr, input_image_hdr)
    # convert image type to float32 (conv function needed)
    input_image_sdr = np.float32(input_image_sdr)
    input_image_hdr = np.float32(input_image_hdr)
    # normalize image and label to [0, 1]
    input_image_sdr = map_range(input_image_sdr, 'sdr')
    input_image_hdr = map_range(input_image_hdr, 'hdr') 
    # crop image and label to [256, 256, 3]
    in_row_ind = random.randint(0, (input_image_sdr.shape[0] - patch_size))
    in_col_ind = random.randint(0, (input_image_sdr.shape[1] - patch_size))

    input_sdr_cropped = input_image_sdr[in_row_ind:in_row_ind + patch_size,
                                        in_col_ind:in_col_ind + patch_size]
    input_hdr_cropped = input_image_hdr[in_row_ind:in_row_ind + patch_size,
                                        in_col_ind:in_col_ind + patch_size]
    
    return (True,input_sdr_cropped,input_hdr_cropped)

def remove_edge(path, type):
    frames =  sorted(glob.glob(path + "/*"))
    num = len(frames)
    for i in range(num):
        if type == 'sdr':
            img = cv2.imread(frames[i])
            img = img[156:924,:,:]
            new_path = 'new_sdr/'+frames[i]
            cv2.imwrite(new_path, img)
        else:
            img = cv2.imread(frames[i], flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
            img = img[156:924,:,:]
            img = img.astype(np.uint16)
            new_path = 'new_hdr/'+frames[i]
            cv2.imwrite(new_path, img)

