from skimage.util.shape import view_as_blocks
import os
import cv2 

import sys 
import numpy as np 
from models.ssd_mobilenet import ssd_300
import cv2
import numpy as np
from keras.optimizers import Adam, SGD
from misc.keras_ssd_loss import SSDLoss 
import os
import h5py
import keras
import time 
from keras import backend as K 
from matplotlib import pyplot as plt 

from scipy.misc import imread    
from keras.preprocessing import image 


from PIL import Image

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
 
import cv2
import time 
import math
from models.mobilenetv2ssd512_last import SSD
from bounding_box_utils.bounding_box_utils import iou # overlaps
import itertools
from operator import itemgetter

### predict with splitting image
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
fileDir = os.path.dirname(os.path.realpath(__file__))


img_height = 512 # Height of the model input images
img_width = 512# Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 43 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales_traffic_sign = [0.04, 0.112, 0.184, 0.256, 0.328, 0.4, 0.472]
scales = scales_pascal
# scales = scales_traffic_sign
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters

 
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
 
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_last_epoch-92_loss-4.6911_val_loss-7.8811.h5') #best weight
# weight_path = os.path.join(rootDir, 'inference/m2ssd/training_mobilenetv2ssd512_class_weight_usetraintest_last_epoch-91_loss-5.5499_val_loss-8.2515.h5') #best
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-125_loss-4.6402_val_loss-7.7933.h5')
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-105_loss-4.6425_val_loss-7.8029.h5')
# weight_path = os.path.join(fileDir, 'training_ssd416_mobilenet_epoch-41_loss-6.0904_val_loss1-0.3212.h5')


## odd scales
# weight_path = os.path.join(rootDir, 'inference/m2ssdoddscales/training_mobilenetv2ssd512_last_odd_scales_last_epoch-108_loss-4.9453_val_loss-7.8760.h5')
weight_path = os.path.join(rootDir, 'inference/m2ssdoodscales_splittingimage/training_mobilenetv2ssd512_odd_scales_splitting_image_epoch-92_loss-4.3373_val_loss-6.6256.h5')

## new scales
# weight_path = os.path.join(rootDir, 'inference/m2ssdnewscales/training_mobilenetv2ssd512_last_new_scales_last_epoch-120_loss-4.6292_val_loss-9.0608.h5')
 


model.load_weights(weight_path) 

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss, metrics=['accuracy'])
dir_path = os.path.join(fileDir, 'test1/') 
fileDir = os.path.dirname(os.path.realpath(__file__))
test = os.path.join(fileDir, 'test1/00001.jpg')
 
# ### inference time
# image = cv2.imread(test)
# tt_arr = []
# predict_arr = []
# decode_arr = []
# for i in range(100):
#     start_time = time.time()
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
#     H, W, _ = image.shape
#     stride = 400
#     size = 512
#     imgs = []
#     input_size = []
#     count_i = 0
#     count_j = 0
#     for i in range(0, H, stride):
#         count_j += 1
#         count_i = 0
#         for j in range(0, W, stride): 
#             count_i += 1
#             img = image[i: min(i + size, H), j: min(j + size, W)]   
#             imgs.append(img)
#             input_size.append((i, j, count_i, count_j)) 
#     gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}
#     convert_to_3_channels = ConvertTo3Channels()
#     resize = Resize(height=img_height,width=img_width, labels_format=gt_format)

#     transformations = [convert_to_3_channels, resize]
#     input_image = [] 
#     trans_imgs = []
#     for img in imgs:   
#         for transform in transformations:
#             img = transform(img)   
#         input_image.append(img) 
#         trans_imgs.append(img)
#     input_image = np.array(input_image) 
#     s = time.time()
#     imgs_predicted = model.predict(input_image)
#     predict_arr.append(time.time() - s)
#     s_decode = time.time()
#     imgs_predicted = decode_detections(imgs_predicted, 
#                             confidence_thresh=0.45,
#                             iou_threshold=0.5,
#                             top_k=100,
#                             input_coords='centroids',
#                             normalize_coords=True,
#                             img_height=img_height,
#                             img_width=img_width)

#     decode_arr.append(time.time() - s_decode)

#     ypred_filtered = [] 
#     for i in range(len(input_image)):
#         y, x, c_i, c_j = input_size[i] 
#         for pred in imgs_predicted[i]:
#             class_id, conf, xmin, ymin, xmax, ymax = pred 
#             # tinh lai xmin, ymin, xmax, ymax tren anh goc
#             xmin = xmin * imgs[i].shape[1] / img_width  
#             ymin = ymin * imgs[i].shape[0] / img_height
#             xmax = xmax * imgs[i].shape[1] / img_width
#             ymax = ymax * imgs[i].shape[0] / img_height 
#             if x != 0 and x!=W:
#                 xmin += (c_i - 1) * stride
#                 xmax += (c_i - 1) * stride
#             if y != 0 and y!=H:
#                 ymin += (c_j - 1) * stride
#                 ymax += (c_j - 1) * stride
#             ypred_filtered.append([class_id, conf, xmin, ymin, xmax, ymax])  

#     tt_arr.append(time.time() - start_time)
#     print ("time taken by ssd", time.time() - start_time)

# # print(tt_arr)
# t = sum(tt_arr[1:]) / len(tt_arr[1:])
# print('done in: ', t)
# print('FPS: ', 1 / t)

# t_predict = sum(predict_arr[1:]) / len(predict_arr[1:])
# t_decode = sum(decode_arr[1:]) / len(decode_arr[1:])
# print('t_predict: ', t_predict)
# print('t_decode: ', t_decode)


## show image
for file in os.listdir(dir_path): 
    filename = dir_path +  file
    print(filename)
    image = cv2.imread(filename) 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
        

    H, W, _ = image.shape
    stride = 400
    size = 512
    imgs = []
    input_size = []
    count_i = 0
    count_j = 0
    for i in range(0, H, stride):
        count_j += 1
        count_i = 0
        for j in range(0, W, stride): 
            count_i += 1
            img = image[i: min(i + size, H), j: min(j + size, W)]   
            imgs.append(img)
            input_size.append((i, j, count_i, count_j))

    print(len(imgs))
    # for i in range(len(imgs)):
    #     img = imgs[i]
    #     y, x, _, _ = input_size[i]
    #     cv2.imshow('img' + str(x) + '_' + str(y), img)
    #     cv2.waitKey(0) 
    # h, w = H//2, W//2
    # imgs = [image[x:x+h,y:y+w] for x in range(0,H, h) for y in range(0,W, w)] 
    gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height,width=img_width, labels_format=gt_format)

    transformations = [convert_to_3_channels, resize]
    input_image = [] 
    trans_imgs = []
    for img in imgs:   
        for transform in transformations:
            img = transform(img)   
        input_image.append(img) 
        trans_imgs.append(img)
    input_image = np.array(input_image) 
    imgs_predicted = model.predict(input_image) # co len(imgs_predicted) = len(imgs)
    imgs_predicted = decode_detections(imgs_predicted, 
                            confidence_thresh=0.45,
                            iou_threshold=0.5,
                            top_k=100,
                            input_coords='centroids',
                            normalize_coords=True,
                            img_height=img_height,
                            img_width=img_width) # moi imgs_predicted[i] co shape(100, 6)

    ypred_filtered = [] 
    for i in range(len(input_image)):
        y, x, c_i, c_j = input_size[i]
        # img = trans_imgs[i]
        for pred in imgs_predicted[i]:
            class_id, conf, xmin, ymin, xmax, ymax = pred
            # print(class_id, conf, y, x, c_i, c_j) 
            # print(xmin, ymin, xmax, ymax)
            # cv2.rectangle(img,(xmin, ymin), (xmax, ymax),(0,255,255),2)
            # cv2.putText(img, str(int(class_id)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0),2) 
            # cv2.imshow('img', img)
            # cv2.waitKey(0) 
            # tinh lai xmin, ymin, xmax, ymax tren anh goc
            xmin = xmin * imgs[i].shape[1] / img_width  
            ymin = ymin * imgs[i].shape[0] / img_height
            xmax = xmax * imgs[i].shape[1] / img_width
            ymax = ymax * imgs[i].shape[0] / img_height
            # print(xmin, ymin, xmax, ymax)
            if x != 0 and x!=W:
                xmin += (c_i - 1) * stride
                xmax += (c_i - 1) * stride
            if y != 0 and y!=H:
                ymin += (c_j - 1) * stride
                ymax += (c_j - 1) * stride
            xmax = int(xmax)
            xmin = int(xmin)
            ymax = int(ymax)
            ymin = int(ymin)
            # print(xmin, ymin, xmax, ymax)
            ypred_filtered.append([class_id, conf, xmin, ymin, xmax, ymax])  
            # cv2.rectangle(image,(xmin, ymin), (xmax, ymax),(0,255,255),2)
            # cv2.putText(image, str(int(class_id)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0),2) 
    # print(ypred_filtered)
    y_pred_group = [list(item[1]) for item in itertools.groupby(sorted(ypred_filtered), key=lambda x: x[0])]
    # print('y pred group: ', y_pred_group)
    # print('len y pred group: ', len(y_pred_group))
    matching_iou_threshold = 0.5
    result = []
    for i in range(len(y_pred_group)):
        ypg = y_pred_group[i]
        if len(ypg) > 1: 
            bb_max = max(enumerate(ypg), key=itemgetter(1))[1]
            bb_max1 = bb_max[2:]
            bb_max1 = np.asarray(bb_max1)
            ypg1 = np.asarray(ypg)
            ypg1 = ypg1[:,2:] 
            overlaps = iou(ypg1, bb_max1, coords='corners',mode='element-wise',border_pixels='include')
            print('overlaps: ', overlaps)
            index_remove = []  
            for j in range(len(overlaps)):
                if overlaps[j] > matching_iou_threshold:
                    index_remove.append(j)

            ypg = [ypg[j] for j in range(len(ypg)) if j not in index_remove]
            if len(ypg) > 0:
                ypg = ypg
                ypg.append(bb_max)
            else:
                ypg = [bb_max] 

            print('ypg: ', ypg)
        result.append(ypg)

    print('result: ', result)
    for r in result:
        print('r: ', r)
        print(len(r))
        for bb in r:
            class_id, conf, xmin, ymin, xmax, ymax = bb
            cv2.rectangle(image,(xmin, ymin), (xmax, ymax),(0,255,255),2)
            cv2.putText(image, str(int(class_id)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0),2) 
    cv2.imshow('img', image)
    cv2.waitKey(0) 

# ypg = [[29.0, 0.5727816820144653, 795, 263, 864, 328], [29.0, 0.7943289875984192, 798, 262, 866, 329]]
# bb_max = [29.0, 0.7943289875984192, 798, 262, 866, 329]

# bb_max = bb_max[2:]
# ypg1 = np.asarray(ypg)
# bb_max = np.asarray(bb_max)
# ypg1 = ypg1[:, 2:]  
# matching_iou_threshold = 0.5
# overlaps = iou(ypg1, bb_max, coords='corners',mode='element-wise',border_pixels='include')
# index_remove = [] 
# for i in range(len(overlaps)):
#     if overlaps[i] > matching_iou_threshold:
#         index_remove.append(i)
        
# ypg = [ypg[i] for i in range(len(ypg)) if i not in index_remove]
# print(ypg)
### training
# def checkBB(gt, tile, size, stride):
#     class_id, xmin, ymin, xmax, ymax = gt
#     h, w, _ = size
#     i, j, c_y, c_x = tile 

#     # tinh toa do goc cua tile
#     x_o = (c_x - 1) * stride
#     y_o = (c_y - 1) * stride
#     w_o = x_o + w
#     h_o = y_o + h
#     if x_o <= xmin and xmax <= w_o and y_o <= ymin and ymax <= h_o:
#         return True
#     return False

# def findBB(gt, tile, stride):
#     class_id, xmin, ymin, xmax, ymax = gt
#     x, y, c_y, c_x= tile

#     # tinh lai toa do bounding box tren tile
#     xmin = xmin - (c_x - 1) * stride
#     xmax = xmax - (c_x - 1) * stride
#     ymin = ymin - (c_y - 1) * stride
#     ymax = ymax - (c_y - 1) * stride
#     return class_id, xmin, ymin, xmax, ymax

# ### training with splitting image
# img_height = 512
# img_width = 512
# fileDir = os.path.dirname(os.path.realpath(__file__))
# images_dir = '/home/ubuntu/Documents/datn/ssd_keras/datasets/full'
# image_dir = os.path.join(fileDir, 'test1')
# test = os.path.join(image_dir, '00001.jpg')
# train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)  
 
# train_labels_filename = os.path.join(rootDir, 'training/train_new.csv') 
# images_t, filenames_t, labels_t, image_ids_t = train_dataset.parse_csv(images_dir=images_dir,
#                         labels_filename=train_labels_filename,
#                         input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
#                         include_classes='all', ret = True)

# for k in range(1):
# # for k in range(len(filenames_t)): 
#     # image = cv2.imread(filenames_t[k])
#     image = cv2.imread(test)
#     H, W, _ = image.shape
#     stride = 400
#     size = 512
#     imgs = []
#     input_size = []
#     tiles = []
#     count_x = 0
#     count_y = 0
#     M = 0
#     N = 0
#     for i in range(0, H, stride):
#         count_y += 1
#         count_x = 0
#         M += 1
#         N = 0
#         for j in range(0, W, stride): 
#             count_x += 1
#             N += 1
#             img = image[i: min(i + size, H), j: min(j + size, W)]   
#             imgs.append(img)
#             input_size.append((j, i, count_y, count_x))
#     print(len(imgs))

#     # labels =  [[  14,  973,  335, 1031,  390],
#     #     [  39,  386,  494,  442,  552],
#     #     [  41,  983,  388, 1024,  432]]

#     # labels = [[  5, 742, 443, 765, 466],
#     #        [ 10, 742, 466, 764, 489],
#     #        [ 22, 737, 412, 769, 443]]
#     labels = list(labels_t[k])
#     print(labels)
    
#     tiles_bbs = []
#     for i in range(M): 
#         tiles_bbs.append([])
#         for j in range(N):
#             tiles_bbs[i].append([])

#     # for i in range(len(imgs)):
#     #     image = imgs[i]   
#     #     i, j, count_i, count_j = input_size[i] 
#         # print(count_i - 1, count_j - 1)
#         # cv2.imshow('image', image)
#         # cv2.waitKey(0)

#     # print(tiles_bbs) 
#     # # tiles_bbs = np.array(tiles_bbs)   
#     # ### tim lai bounding box cua object 
#     # for bb in labels:
#     #     class_id, xmin, ymin, xmax, ymax = bb
#     #     x_min_stride = xmin // stride
#     #     x_max_stride = xmax // img_width
#     #     y_min_stride = ymin // stride
#     #     y_max_stride = ymax // img_height
#     #     print(class_id, xmin, ymin, xmax, ymax)
#     #     print(x_min_stride, x_max_stride, y_min_stride, y_max_stride)
#     #     tx = -1
#     #     ty = -1
#     #     if x_min_stride == x_max_stride:
#     #         tx = x_min_stride
#     #     if y_min_stride == y_max_stride:
#     #         ty = y_max_stride
#     #     print(tx, ty) 
#     #     if tx > -1 and ty > -1 and tx * stride < xmin < (tx + 1) * img_width and xmax < (tx + 1) * img_width \
#     #         and ty * stride < ymin < (ty + 1) * img_height and ymax < (ty + 1) * img_height:
#     #         xmin_tile = xmin - tx * stride
#     #         xmax_tile = xmax - tx * stride
#     #         ymin_tile = ymin - ty * stride
#     #         ymax_tile = ymax - ty * stride
#     #         print(xmin_tile, ymin_tile, xmax_tile, ymax_tile)
#     #         tiles_bbs[ty][tx].append([class_id, xmin_tile, ymin_tile, xmax_tile, ymax_tile])
        

#     # print(tiles_bbs)
#     batch_y = []
#     for i in range(len(imgs)):
#         tile_bbs = []
#         x, y, c_y, c_x = input_size[i]
#         c_y -= 1
#         c_x -= 1
#         print(c_y, c_x)
#         for gt in labels:
#             if checkBB(gt, input_size[i], imgs[i].shape, stride):
#                 class_id, xmin, ymin, xmax, ymax = findBB(gt, input_size[i], stride)
#                 tile_bbs.append([class_id, xmin, ymin, xmax, ymax])

#                 # tiles_bbs[c_y][c_x].append([class_id, xmin, ymin, xmax, ymax])
#         #         xmax = int(xmax)
#         #         xmin = int(xmin)
#         #         ymax = int(ymax)
#         #         ymin = int(ymin)
#         #         cv2.rectangle(imgs[i],(xmin, ymin), (xmax, ymax),(0,255,255),2)
#         #         cv2.putText(imgs[i], str(int(class_id)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0),2) 

#         # cv2.imshow('img', imgs[i])
#         # cv2.waitKey(0) 

#             # print(checkBB(gt, input_size[i], imgs[i].shape, stride), input_size[i])
#         batch_y.append(np.array(tile_bbs))
#     print(batch_y)

# ### tai sao su dung size = 512
# ### tai sao su dung stride = 400
# ### thay the split tu split_overlap sang data_generator
# ### chu y den batch_X, batch_y    