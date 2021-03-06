from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
import glob
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)
import imageio
# DAVIS

# # User defined parameters
# seq_name = "car-shadow"
# gpu_id = 0
# train_model = True
# result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
#
# # Train parameters
# parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
# logs_path = os.path.join('models', seq_name)
# max_training_iters = 500
#
# # Define Dataset
# test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
# test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
# if train_model:
#     train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
#                   os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
#     dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
# else:
#     dataset = Dataset(None, test_imgs, './')

# smoke dataset

# User defined parameters
seq_name = "wildfire_smoke_3"
gpu_id = 0
train_model = False
result_path = os.path.join('dataset', 'test_dataset', seq_name, 'Results')

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-9000')
logs_path = os.path.join('models', seq_name)
if not os.path.exists(logs_path):
    os.mkdir(logs_path)
max_training_iters = 500





# Define Dataset
import re
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval
def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
# test_frames = sorted(os.listdir(os.path.join('dataset', 'smoke_dataset', seq_name)))
test_imgs = glob.glob(os.path.join('dataset','test_dataset',seq_name,'Image', '*.jpg'))
test_flows = glob.glob(os.path.join('dataset','test_dataset',seq_name,'Flow', '*.npy'))
test_imgs.sort(key=natural_keys)
test_flows.sort(key=natural_keys)
final_test_list = [(x,y) for x,y in zip(test_imgs, test_flows)]
print(final_test_list)
# test_imgs = [os.path.join('dataset', 'smoke_dataset', seq_name, frame) for frame in test_frames]
if train_model:
    train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                  os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
    dataset = Dataset(train_imgs, final_test_list, './', data_aug=True)
else:
    dataset = Dataset(None, final_test_list, './dataset/test_dataset/*/', flow_given=True)

# Train the network
if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name, backbone='resnet')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        # checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        checkpoint_path = parent_path
        osvos.test(dataset, checkpoint_path, result_path, backbone='vgg')

# Show results
overlay_color = [255, 0, 0]
transparency = 0.6
plt.ion()
for img_p in test_imgs:
    frame_num = os.path.basename(img_p).split('.')[0]
    img = np.array(Image.open(img_p))
    mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
    mask = mask//np.max(mask)
    mask = 1 - mask
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
    new_result_path = "/data1/taanya1/new_ICPR/ICPR/dataset/test_dataset/wildfire_smoke_3/Flow_rgb/"
    imageio.imwrite(os.path.join(new_result_path, frame_num+'.png'), im_over.astype(np.uint8))  
    plt.imshow(im_over.astype(np.uint8))
    plt.axis('off')
    plt.show()
    plt.pause(0.01)
    plt.clf()
