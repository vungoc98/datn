# '''
# A custom Keras layer to decode the raw SSD prediction output. Corresponds to the
# `DetectionOutput` layer type in the original Caffe implementation of SSD.

# Copyright (C) 2018 Pierluigi Ferrari

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# '''

# from __future__ import division
# import numpy as np
# import tensorflow as tf
# import keras.backend as K
# from keras.engine.topology import InputSpec
# from keras.engine.topology import Layer

# class DecodeDetections(Layer):
#     '''
#     A Keras layer to decode the raw SSD prediction output.

#     Input shape:
#         3D tensor of shape `(batch_size, n_boxes, n_classes + 12)`.

#     Output shape:
#         3D tensor of shape `(batch_size, top_k, 6)`.
#     '''

#     def __init__(self,
#                  confidence_thresh=0.01,
#                  iou_threshold=0.45,
#                  top_k=200,
#                  nms_max_output_size=400,
#                  coords='centroids',
#                  normalize_coords=True,
#                  img_height=None,
#                  img_width=None,
#                  **kwargs):
#         '''
#         All default argument values follow the Caffe implementation.

#         Arguments:
#             confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
#                 positive class in order to be considered for the non-maximum suppression stage for the respective class.
#                 A lower value will result in a larger part of the selection process being done by the non-maximum suppression
#                 stage, while a larger value will result in a larger part of the selection process happening in the confidence
#                 thresholding stage.
#             iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
#                 with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
#                 to the box score.
#             top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
#                 non-maximum suppression stage.
#             nms_max_output_size (int, optional): The maximum number of predictions that will be left after performing non-maximum
#                 suppression.
#             coords (str, optional): The box coordinate format that the model outputs. Must be 'centroids'
#                 i.e. the format `(cx, cy, w, h)` (box center coordinates, width, and height). Other coordinate formats are
#                 currently not supported.
#             normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
#                 and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
#                 relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
#                 Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
#                 coordinates. Requires `img_height` and `img_width` if set to `True`.
#             img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
#             img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
#         '''
#         if K.backend() != 'tensorflow':
#             raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

#         if normalize_coords and ((img_height is None) or (img_width is None)):
#             raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

#         if coords != 'centroids':
#             raise ValueError("The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

#         # We need these members for the config.
#         self.confidence_thresh = confidence_thresh
#         self.iou_threshold = iou_threshold
#         self.top_k = top_k
#         self.normalize_coords = normalize_coords
#         self.img_height = img_height
#         self.img_width = img_width
#         self.coords = coords
#         self.nms_max_output_size = nms_max_output_size

#         # We need these members for TensorFlow.
#         self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
#         self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
#         self.tf_top_k = tf.constant(self.top_k, name='top_k')
#         self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
#         self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
#         self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
#         self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

#         super(DecodeDetections, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.input_spec = [InputSpec(shape=input_shape)]
#         super(DecodeDetections, self).build(input_shape)

#     def call(self, y_pred, mask=None):
#         '''
#         Returns:
#             3D tensor of shape `(batch_size, top_k, 6)`. The second axis is zero-padded
#             to always yield `top_k` predictions per batch item. The last axis contains
#             the coordinates for each predicted box in the format
#             `[class_id, confidence, xmin, ymin, xmax, ymax]`.
#         '''

#         #####################################################################################
#         # 1. Convert the box coordinates from predicted anchor box offsets to predicted
#         #    absolute coordinates
#         #####################################################################################

#         # Convert anchor box offsets to image offsets.
#         cx = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
#         cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
#         w = tf.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6] # w = exp(w_pred * variance_w) * w_anchor
#         h = tf.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5] # h = exp(h_pred * variance_h) * h_anchor

#         # Convert 'centroids' to 'corners'.
#         xmin = cx - 0.5 * w
#         ymin = cy - 0.5 * h
#         xmax = cx + 0.5 * w
#         ymax = cy + 0.5 * h

#         # If the model predicts box coordinates relative to the image dimensions and they are supposed
#         # to be converted back to absolute coordinates, do that.
#         def normalized_coords():
#             xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
#             ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
#             xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
#             ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
#             return xmin1, ymin1, xmax1, ymax1
#         def non_normalized_coords():
#             return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

#         xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

#         # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
#         y_pred = tf.concat(values=[y_pred[...,:-12], xmin, ymin, xmax, ymax], axis=-1)

#         #####################################################################################
#         # 2. Perform confidence thresholding, per-class non-maximum suppression, and
#         #    top-k filtering.
#         #####################################################################################

#         batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
#         n_boxes = tf.shape(y_pred)[1]
#         n_classes = y_pred.shape[2] - 4 
#         class_indices = tf.range(1, n_classes)

#         # Create a function that filters the predictions for the given batch item. Specifically, it performs:
#         # - confidence thresholding
#         # - non-maximum suppression (NMS)
#         # - top-k filtering
#         def filter_predictions(batch_item): 
#             # Keep only the non-background boxes. 
            
#             # Create a function that filters the predictions for one single class.
#             def filter_single_class(index):

#                 # From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract
#                 # a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
#                 # confidnece values for just one class, determined by `index`.
#                 confidences = tf.expand_dims(batch_item[..., index], axis=-1)
#                 class_id = tf.fill(dims=tf.shape(confidences), value=tf.to_float(index))
#                 box_coordinates = batch_item[...,-4:]

#                 single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)

#                 # Apply confidence thresholding with respect to the class defined by `index`.
#                 threshold_met = single_class[:,1] > self.tf_confidence_thresh
#                 single_class = tf.boolean_mask(tensor=single_class,
#                                                mask=threshold_met)

#                 # If any boxes made the threshold, perform NMS.
#                 def perform_nms():
#                     scores = single_class[...,1]

#                     # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
#                     xmin = tf.expand_dims(single_class[...,-4], axis=-1)
#                     ymin = tf.expand_dims(single_class[...,-3], axis=-1)
#                     xmax = tf.expand_dims(single_class[...,-2], axis=-1)
#                     ymax = tf.expand_dims(single_class[...,-1], axis=-1)
#                     boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

#                     maxima_indices = tf.image.non_max_suppression(boxes=boxes,
#                                                                   scores=scores,
#                                                                   max_output_size=self.tf_nms_max_output_size,
#                                                                   iou_threshold=self.iou_threshold,
#                                                                   name='non_maximum_suppresion')
#                     maxima = tf.gather(params=single_class,
#                                        indices=maxima_indices,
#                                        axis=0)
#                     return maxima

#                 def no_confident_predictions():
#                     return tf.constant(value=0.0, shape=(1,6))

#                 single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

#                 # Make sure `single_class` is exactly `self.nms_max_output_size` elements long.
#                 padded_single_class = tf.pad(tensor=single_class_nms,
#                                              paddings=[[0, self.tf_nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
#                                              mode='CONSTANT',
#                                              constant_values=0.0)

#                 return padded_single_class

#             # Iterate `filter_single_class()` over all class indices.
#             filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
#                                                 elems=tf.range(1,n_classes),
#                                                 dtype=tf.float32,
#                                                 parallel_iterations=128,
#                                                 back_prop=False,
#                                                 swap_memory=False,
#                                                 infer_shape=True,
#                                                 name='loop_over_classes')

#             # Concatenate the filtered results for all individual classes to one tensor.
#             filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1,6))

#             # Perform top-k filtering for this batch item or pad it in case there are
#             # fewer than `self.top_k` boxes left at this point. Either way, produce a
#             # tensor of length `self.top_k`. By the time we return the final results tensor
#             # for the whole batch, all batch items must have the same number of predicted
#             # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
#             # predictions are left after the filtering process above, we pad the missing
#             # predictions with zeros as dummy entries.
#             def top_k():
#                 return tf.gather(params=filtered_predictions,
#                                  indices=tf.nn.top_k(filtered_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
#                                  axis=0)
#             def pad_and_top_k():
#                 padded_predictions = tf.pad(tensor=filtered_predictions,
#                                             paddings=[[0, self.tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
#                                             mode='CONSTANT',
#                                             constant_values=0.0)
#                 return tf.gather(params=padded_predictions,
#                                  indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
#                                  axis=0)

#             top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], self.tf_top_k), top_k, pad_and_top_k)

#             return top_k_boxes


#         # Iterate `filter_predictions()` over all batch items.
#         output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
#                                   elems=y_pred,
#                                   dtype=None,
#                                   parallel_iterations=128,
#                                   back_prop=False,
#                                   swap_memory=False,
#                                   infer_shape=True,
#                                   name='loop_over_batch')

#         return output_tensor

#     def compute_output_shape(self, input_shape):
#         batch_size, n_boxes, last_axis = input_shape
#         return (batch_size, self.tf_top_k, 6) # Last axis: (class_ID, confidence, 4 box coordinates)

#     def get_config(self):
#         config = {
#             'confidence_thresh': self.confidence_thresh,
#             'iou_threshold': self.iou_threshold,
#             'top_k': self.top_k,
#             'nms_max_output_size': self.nms_max_output_size,
#             'coords': self.coords,
#             'normalize_coords': self.normalize_coords,
#             'img_height': self.img_height,
#             'img_width': self.img_width,
#         }
#         base_config = super(DecodeDetections, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))




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

from models.mobilenet_v2 import SSD
import cv2
import time
import threading
# from models.mobilenet_v2_lite import SSD
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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

# aspect_ratios = [[1.0, 1.0],
#                  [1.0, 1.0, 1.0],
#                  [1.0, 1.0, 1.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
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

# # weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-51_loss-9.8239_val_loss-10.3842.h5')
# # weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-34_loss-9.4036_val_loss-8.7010.h5')
# weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-74_loss-9.7366_val_loss-9.6044.h5')
# weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-98_loss-4.4158_val_loss-7.3604.h5')

# weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet_epoch-19_loss-5.2135_val_loss-6.6133.h5')

# weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet_new_classID_epoch-21_loss-5.1539_val_loss-6.3321.h5')
# weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet_new_classID_epoch-44_loss-4.3979_val_loss-5.1283.h5')

# weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet_new_classID_epoch-95_loss-3.6799_val_loss-4.6639.h5')
# weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet-allDATA-newAUG_epoch-86_loss-4.8915_val_loss-7.9571.h5')
weight_path = os.path.join(fileDir, 'training_ssd512_mobilenet-allDATA-newAUG_epoch-96_loss-4.7865_val_loss-7.9207.h5')
# weight_path = os.path.join(rootDir, 'training/training_ssd512_mobilenet_lite_epoch-10_loss-8.8584_val_loss-53.4644.h5')
model.load_weights(weight_path) 

# model.load_weights(os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-16_loss-6.4981_val_loss-8.0133.h5'), by_name=True,skip_mismatch=True)
# model.load_weights(os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-67_loss-4.8934_val_loss-7.1338.h5'))

# We'll only load one image in this example.
img_path = os.path.join(fileDir, '00801.jpg')
tt_arr = []
# original_image = imread(img_path)  
from threading import Lock
l = Lock()

def th(image1):
    global l
    y_pred = model.predict(image1) 
    confidence_threshold = 0.45
    y_pred_decoded = decode_detections(y_pred, 
                                        confidence_thresh=0.45,
                                        iou_threshold=0.5,
                                        top_k=10,
                                        input_coords='centroids',
                                        normalize_coords=True,
                                        img_height=img_height,
                                        img_width=img_width)
 
    for box in y_pred_decoded[0]:

        xmin = int(box[2] * orig_images[0].shape[1] / img_width)
        ymin = int(box[3] * orig_images[0].shape[0] / img_height)
        xmax = int(box[4] * orig_images[0].shape[1] / img_width)
        ymax = int(box[5] * orig_images[0].shape[0] / img_height)
    l.acquire()
    tt_arr.append(time.time() - start_time)
    l.release()
    print ("time taken by ssd", time.time() - start_time) 

img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
for tt in range(100):  
    start_time = time.time()  
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.
    orig_images.append(img) 
 
    image1 = cv2.resize(img,(img_height,img_width))
    image1 = np.array(image1,dtype=np.float32) 

    image1 = image1[np.newaxis,:,:,:]
    t = threading.Thread(target=th, args=(image1, ))
    t.start()
    # input_images.append(image1)
    # input_images = np.array(image1)
 
 
    # y_pred = model.predict(image1) 
    # confidence_threshold = 0.45
    # y_pred_decoded = decode_detections(y_pred, 
    #                                     confidence_thresh=0.45,
    #                                     iou_threshold=0.5,
    #                                     top_k=10,
    #                                     input_coords='centroids',
    #                                     normalize_coords=True,
    #                                     img_height=img_height,
    #                                     img_width=img_width)
 
    # for box in y_pred_decoded[0]:

    #     xmin = int(box[2] * orig_images[0].shape[1] / img_width)
    #     ymin = int(box[3] * orig_images[0].shape[0] / img_height)
    #     xmax = int(box[4] * orig_images[0].shape[1] / img_width)
    #     ymax = int(box[5] * orig_images[0].shape[0] / img_height)
   
    # tt_arr.append(time.time() - start_time)
    # print ("time taken by ssd", time.time() - start_time) 




# print(tt_arr)
t =  sum(tt_arr[1:]) / len(tt_arr[1:])
print('done in: ', t)
print('FPS: ', 1 / t)
