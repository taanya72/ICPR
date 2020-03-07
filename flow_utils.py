import os
import numpy as np
from PIL import Image
import time
import argparse
import re
import glob
from pyflow import pyflow

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def glob_files(path):
    files = []
    for ext in ('*.jpeg', '*.png', '*.jpg', '*.bmp'):
        files.extend(glob.glob(os.path.join(path, ext)))
    return files

def run_flow(dir_path):
    '''

    Args:
        dir_path: it should be like '/dataset/smoke_dataset/wildfire_smoke_1

    Returns:

    '''
    print("run dense flow for images in {}".format(dir_path))
    # get the images
    images_dir = os.path.join(dir_path, 'Image')
    assert os.path.exists(images_dir), "the image foler {} does not exist.".format(images_dir)
    output_dir = os.path.join(dir_path, 'Flow')
    image_paths = glob_files(images_dir)
    image_paths.sort(key=natural_keys)
    num_images = len(image_paths)
    # pad first and last images by copying the data
    padded_image_paths = [image_paths[0]] + image_paths + [image_paths[-1]]
    for i in range(num_images):
        prev_image_path, next_image_path = padded_image_paths[i], padded_image_paths[i+1]
        pre_image = np.array(Image.open(prev_image_path), dtype=np.double) / 255.
        next_image = np.array(Image.open(next_image_path), dtype=np.double) / 255.
        print(pre_image)
        u, v, im2W = pyflow.coarse2fine_flow(
            pre_image, next_image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        np.save(os.path.join(output_dir, 'outFlow' + str(i) + '.npy'), flow)
        # im1 = np.array(Image.open(im1_path))
        # im2 = np.array(Image.open(im2_path))
        # im1 = im1.astype(float) / 255.
        # im2 = im2.astype(float) / 255.



