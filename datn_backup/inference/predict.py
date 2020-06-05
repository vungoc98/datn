import sys
sys.path.append("/home/manish/MobileNet-ssd-keras")
import numpy as np 
from models.ssd_mobilenet import ssd_300
import cv2
import numpy as np
from keras.optimizers import Adam, SGD
from misc.keras_ssd_loss import SSDLoss
# from misc.
import os
import h5py
import keras
import time
from keras.preprocessing import image
from misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from keras import backend as K
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from matplotlib import pyplot as plt

# fileDir = os.path.dirname(os.path.realpath(__file__))

from scipy.misc import imread    
from keras.preprocessing import image

# img_height = 300 # Height of the model input images
# img_width = 300 # Width of the model input images
# img_channels = 3 # Number of color channels of the model input images
# mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
# swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
# n_classes = 43 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# scales_traffic_sign = [0.04, 0.112, 0.184, 0.256, 0.328, 0.4, 0.472]
# scales = scales_pascal
# # scales = scales_traffic_sign
# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters

# # aspect_ratios = [[1.0, 1.0],
# #                  [1.0, 1.0, 1.0],
# #                  [1.0, 1.0, 1.0],
# #                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
# #                  [1.0, 2.0, 0.5],
# #                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
# two_boxes_for_ar1 = True
# steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
# clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
# variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
# normalize_coords = True

# # 1: Build the Keras model.

# K.clear_session() # Clear previous models from memory.

# model = ssd_300(image_size=(img_height, img_width, img_channels),
#                 n_classes=n_classes,
#                 mode='inference',
#                 l2_regularization=0.0005,
#                 scales=scales,
#                 aspect_ratios_per_layer=aspect_ratios,
#                 two_boxes_for_ar1=two_boxes_for_ar1,
#                 steps=steps,
#                 offsets=offsets,
#                 clip_boxes=clip_boxes,
#                 variances=variances,
#                 normalize_coords=normalize_coords,
#                 subtract_mean=mean_color,
#                 swap_channels=swap_channels)

# weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-74_loss-9.7366_val_loss-9.6044.h5')
# model.load_weights(weight_path) 

# dir_path = os.path.join(fileDir, 'test/')


# for file in os.listdir(dir_path):
#     filename = dir_path +  file

#     print (filename)

#     # img = cv2.imread(filename)
#     # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#     # # img1 = ima[90:390,160:460]
#     # img1 = cv2.resize(ima,dsize=(img_height,img_width))
#     # im = img1
#     orig_images = []  # Store the images here.
#     input_images = []  # Store resized versions of the images here.
#     # orig_images.append(img)

#     # img1 = image.img_to_array(img1)
#     # input_images.append(img1)
#     # input_images = np.array(input_images)

#     orig_images.append(imread(img_path))
#     img = image.load_img(img_path, target_size=(img_height, img_width))
#     img = image.img_to_array(img) 
#     input_images.append(img)
#     input_images = np.array(input_images) 

#     # ima = img
#     # # img = img[:,a:a+320]
#     # image1 = cv2.resize(img,(300,300))
#     # image1 = np.array(image1,dtype=np.float32)

#     # image1[:,:,0] = 0.007843*(image1[:,:,0] - 127.5)
#     # image1[:,:,1] = 0.007843*(image1[:,:,1] - 127.5)
#     # image1[:,:,2] = 0.007843*(image1[:,:,2] - 127.5)
#     # image1 = image1[:,:,::-1]

#     # image1 = image1[np.newaxis,:,:,:]
#     # input_images.append(image1)
#     # input_images = np.array(image1)


#     start_time = time.time()


#     y_pred = model.predict(input_images)
#     print(y_pred)
#     # print y_pred.shape
#     # y_pred = y_pred.flatten()
#     # print (y_pred[:15])

#     # print 'y_pred shape', y_pred.shape

#     print ("time taken by ssd", time.time() - start_time)
#     print(max(y_pred[0,:,1]))
#     confidence_threshold = 0.21

#     y_pred_decoded = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]


#     # y_pred_decoded = decode_y(y_pred,
#     #                           confidence_thresh=0.25,
#     #                           iou_threshold=0.45,
#     #                           top_k=100,
#     #                           input_coords='centroids',
#     #                           normalize_coords=True,
#     #                           img_height=img_height,
#     #                           img_width=img_width)


#     # Display the image and draw the predicted boxes onto it.

#     # Set the colors for the bounding boxes
#     colors = plt.cm.hsv(np.linspace(0, 1, 222)).tolist()
#     classes = ['background'] + list(range(1, 222))

#     plt.figure(figsize=(20,12))
#     plt.imshow(orig_images[0])

#     current_axis = plt.gca()

#     for box in y_pred_decoded[0]:
#         # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#         xmin = box[2] * orig_images[0].shape[1] / img_width
#         ymin = box[3] * orig_images[0].shape[0] / img_height
#         xmax = box[4] * orig_images[0].shape[1] / img_width
#         ymax = box[5] * orig_images[0].shape[0] / img_height
#         color = colors[int(box[0])]
#         label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#         current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
#         current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

#     plt.show()

#     # for box in y_pred_decoded[0]:

#     #     xmin = int(box[-4] * orig_images[0].shape[1] / img_width)
#     #     ymin = int(box[-3] * orig_images[0].shape[0] / img_height)
#     #     xmax = int(box[-2] * orig_images[0].shape[1] / img_width)
#     #     ymax = int(box[-1] * orig_images[0].shape[0] / img_height)

#     #     # print int(box[-4]), int(box[-2]) , int(box[-3]) , int(box[-1])
#     #     print (box[0], xmin,xmax,ymin,ymax)
#     #     cv2.rectangle(orig_images[0],(xmin, ymin), (xmax, ymax),(0,255,255),2)
#     #     cv2.putText(orig_images[0], str(int(box[0])), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),2) 



#     # cv2.imshow("image1",orig_images[0])
#     # cv2.waitKey(0)


# import sys
# sys.path.append("/home/manish/MobileNet-ssd-keras")
# import numpy as np 
# from models.ssd_mobilenet import ssd_300
# import cv2
# import numpy as np
# from keras.optimizers import Adam
# from misc.keras_ssd_loss import SSDLoss
# # from misc.
# import os
# import h5py
# import keras
# import time
# from keras.preprocessing import image
# from misc.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
# from keras import backend as K
# # os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# fileDir = os.path.dirname(os.path.realpath(__file__))



# img_height = 300  # Height of the input images
# img_width = 300  # Width of the input images
# img_channels = 3  # Number of color channels of the input images
# subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
# swap_channels = True  # The color channel order in the original SSD is BGR
# n_classes = 6  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
#               1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
#                1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# scales = scales_voc

# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
# two_boxes_for_ar1 = True
# steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
#            0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
# limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
# variances = [0.1, 0.1, 0.2,
#              0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
# coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
# normalize_coords = True

# # 1: Build the Keras model

# # K.clear_session()  # Clear previous models from memory.

# # def train(args):
# model = ssd_300(mode = 'inference',
#                 image_size=(img_height, img_width, img_channels),
#                 n_classes=n_classes,
#                 l2_regularization=0.0005,
#                 scales=scales,
#                 aspect_ratios_per_layer=aspect_ratios,
#                 two_boxes_for_ar1=two_boxes_for_ar1,
#                 steps=steps,
#                 offsets=offsets,
#                 limit_boxes=limit_boxes,
#                 variances=variances,
#                 coords=coords,
#                 normalize_coords=normalize_coords,
#                 subtract_mean=subtract_mean,
#                 divide_by_stddev=None,
#                 swap_channels=swap_channels)

# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

# ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)


# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# print('model.summary: ', model.summary())

# for layer in model.layers:
#     layer.name = layer.name + "_v1"




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
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
fileDir = os.path.dirname(os.path.realpath(__file__))


img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 6 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# scales_traffic_sign = [0.04, 0.112, 0.184, 0.256, 0.328, 0.4, 0.472]
scales = scales_pascal
# scales = scales_traffic_sign
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
steps = None # The space between two adjacent anchor box center points for each predictor layer.
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


print('model.summary: ', model.summary())

# model.load_weights(os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-16_loss-6.4981_val_loss-8.0133.h5'), by_name=True,skip_mismatch=True)
# model.load_weights(os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-67_loss-4.8934_val_loss-7.1338.h5'))

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss, metrics=['accuracy'])


# # weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-51_loss-9.8239_val_loss-10.3842.h5')
# # weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-34_loss-9.4036_val_loss-8.7010.h5')
# weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-74_loss-9.7366_val_loss-9.6044.h5')
weight_path = os.path.join(fileDir, 'training_ssd300_mobilenet_epoch-98_loss-4.4158_val_loss-7.3604.h5')
model.load_weights(weight_path) 

dir_path = os.path.join(fileDir, 'test/')


for file in os.listdir(dir_path):
    filename = dir_path +  file

    print (filename)

    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # # img1 = ima[90:390,160:460]
    # img1 = cv2.resize(ima,dsize=(img_height,img_width))
    # im = img1
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.
    orig_images.append(img)

    # img1 = image.img_to_array(img1)
    # input_images.append(img1)
    # input_images = np.array(input_images)



    ima = img
    # img = img[:,a:a+320]
    image1 = cv2.resize(img,(300,300))
    image1 = np.array(image1,dtype=np.float32)

    # image1[:,:,0] = 0.007843*(image1[:,:,0] - 127.5)
    # image1[:,:,1] = 0.007843*(image1[:,:,1] - 127.5)
    # image1[:,:,2] = 0.007843*(image1[:,:,2] - 127.5)
    # image1 = image1[:,:,::-1]

    image1 = image1[np.newaxis,:,:,:]
    # input_images.append(image1)
    input_images = np.array(image1)


    start_time = time.time()


    y_pred = model.predict(input_images)
    print(y_pred)
    # print y_pred.shape
    # y_pred = y_pred.flatten()
    # print (y_pred[:15])

    # print 'y_pred shape', y_pred.shape

    print ("time taken by ssd", time.time() - start_time)
    print(max(y_pred[0,:,1]))
    confidence_threshold = 0.3

    y_pred_decoded = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]


    # y_pred_decoded = decode_y(y_pred,
    #                           confidence_thresh=0.25,
    #                           iou_threshold=0.45,
    #                           top_k=100,
    #                           input_coords='centroids',
    #                           normalize_coords=True,
    #                           img_height=img_height,
    #                           img_width=img_width)


        

    for box in y_pred_decoded[0]:

        xmin = int(box[2] * orig_images[0].shape[1] / 300)
        ymin = int(box[3] * orig_images[0].shape[0] / 300)
        xmax = int(box[4] * orig_images[0].shape[1] / 300)
        ymax = int(box[5] * orig_images[0].shape[0] / 300)

        # print int(box[-4]), int(box[-2]) , int(box[-3]) , int(box[-1])
        print (xmin,xmax,ymin,ymax)
        cv2.rectangle(orig_images[0],(xmin, ymin), (xmax, ymax),(0,255,255),2)
        cv2.putText(orig_images[0], str(int(box[1])), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),2) 



    cv2.imshow("image1",orig_images[0])
    cv2.waitKey(0)