from scipy.misc import imread    
from keras.preprocessing import image  
import sys 
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

import os
fileDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.join(fileDir, '..')
sys.path.append(rootDir)

# from models.ssd_mobilenet import ssd_300
from misc.keras_ssd_loss import SSDLoss, FocalLoss, weightedSSDLoss, weightedFocalLoss
from misc.keras_layer_AnchorBoxes import AnchorBoxes
from misc.keras_layer_L2Normalization import L2Normalization
from misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from misc.ssd_batch_generator import BatchGenerator
from keras.utils.training_utils import multi_gpu_model
 
import keras
import argparse

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
 
import cv2
import time 
from models.mobilenetv2ssd512_last import SSD
fileDir = os.path.dirname(os.path.realpath(__file__))


img_height = 512 # Height of the model input images
img_width = 512 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 43 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales_traffic_sign = [0.04, 0.112, 0.184, 0.256, 0.328, 0.4, 0.472]
# scales = scales_pascal
scales = scales_traffic_sign
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
 
two_boxes_for_ar1 = True
steps = None# The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
batch_size = 8

# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

model = SSD(input_shape=(img_height, img_width, img_channels),
                num_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)


# weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet-allDATA-newAUG_epoch-86_loss-4.8915_val_loss-7.9571.h5')
# weight_path = os.path.join(rootDir, 'inference/m2ssd/training_mobilenetv2ssd512_class_weight_usetraintest_last_epoch-91_loss-5.5499_val_loss-8.2515.h5')
# weight_path = os.path.join(rootDir, 'training/')

## odd scales
# weight_path = os.path.join(rootDir, 'inference/m2ssdoddscales/training_mobilenetv2ssd512_last_odd_scales_last_epoch-108_loss-4.9453_val_loss-7.8760.h5')


# ## new scales
weight_path = os.path.join(rootDir, 'inference/m2ssdnewscales/training_mobilenetv2ssd512_last_new_scales_last_epoch-120_loss-4.6292_val_loss-9.0608.h5')
 


model.load_weights(weight_path) 
 
# We'll only load one image in this example.
img_path = os.path.join(fileDir, 'test1/00801.jpg')
# img_path = os.path.join(fileDir, '00801.jpg')
tt_arr = []
# original_image = imread(img_path) 
img = cv2.imread(img_path)
for tt in range(100):  
    start_time = time.time() 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.
    orig_images.append(img) 


    ima = img 
    image1 = cv2.resize(img,(img_height,img_width))
    # image1 = np.array(image1,dtype=np.float32) 

    image1 = image1[np.newaxis,:,:,:]
    # input_images.append(image1)
    # input_images = np.array(image1)
 
 
    y_pred = model.predict(image1) 
    confidence_threshold = 0.45
    y_pred_decoded = decode_detections_fast(y_pred, 
                                        confidence_thresh=confidence_threshold,
                                        iou_threshold=0.45,
                                        top_k=100,
                                        input_coords='centroids',
                                        normalize_coords=True,
                                        img_height=img_height,
                                        img_width=img_width)

    # y_pred_decoded = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

 
    for box in y_pred_decoded[0]:

        xmin = int(box[2] * orig_images[0].shape[1] / img_width)
        ymin = int(box[3] * orig_images[0].shape[0] / img_height)
        xmax = int(box[4] * orig_images[0].shape[1] / img_width)
        ymax = int(box[5] * orig_images[0].shape[0] / img_height)
  
    tt_arr.append(time.time() - start_time)
    print ("time taken by ssd", time.time() - start_time)

# print(tt_arr)
t = sum(tt_arr[1:]) / len(tt_arr[1:])
print('done in: ', t)
print('FPS: ', 1 / t)
