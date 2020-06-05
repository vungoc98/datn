# from keras import backend as K
# from keras.models import load_model
# from keras.preprocessing import image
# from keras.optimizers import Adam
# from imageio import imread
# import numpy as np
# from matplotlib import pyplot as plt
  
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
# from keras_layers.keras_layer_L2Normalization import L2Normalization

# from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

# from data_generator.object_detection_2d_data_generator import DataGenerator
# from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
# from data_generator.object_detection_2d_geometric_ops import Resize
# from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
# import cv2

# from PIL import Image

# import operator 
# import time
# from keras.optimizers import Adam, SGD
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
# from keras import backend as K
# from keras.models import load_model
# from math import ceil
# import numpy as np
# import os
# from matplotlib import pyplot as plt  
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
# from keras_layers.keras_layer_L2Normalization import L2Normalization

# from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
# from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

# from data_generator.object_detection_2d_data_generator import DataGenerator
# from data_generator.object_detection_2d_geometric_ops import Resize
# from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
# from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
# from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
# from keras.callbacks import LambdaCallback
# import json

# # from models.mobilenetv2 import SSD
  
# from scipy.misc import imread     
# from eval_utils.average_precision_evaluator import Evaluator
 
# from models.vgg16ssd300_last import SSD
# import cv2
# import time
# # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# fileDir = os.path.dirname(os.path.realpath(__file__))
# rootDir = os.path.join(fileDir, '..')


# img_height = 300 # Height of the model input images
# img_width = 300 # Width of the model input images
# img_channels = 3 # Number of color channels of the model input images
# mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
# swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
# n_classes = 43 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# scales_traffic_sign = [0.04, 0.112, 0.184, 0.256, 0.328, 0.4, 0.472]
# # scales = scales_pascal
# scales = scales_traffic_sign
# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters 
# two_boxes_for_ar1 = True
# steps = None # The space between two adjacent anchor box center points for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
# clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
# variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
# normalize_coords = True
# batch_size = 8

# # 1: Build the Keras model.

# K.clear_session() # Clear previous models from memory.

# model = SSD(image_size=(img_height, img_width, img_channels),
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

# weight_path = os.path.join(rootDir, 'inference/vgg16/training_vgg16ssd300_last_epoch-92_loss-5.4329_val_loss-10.3008.h5')
# model.load_weights(weight_path)  

import sys 
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
from keras import backend as K 
from matplotlib import pyplot as plt

# fileDir = os.path.dirname(os.path.realpath(__file__))

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
# from models.mobilenetv2ssd512_last import SSD
# # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# fileDir = os.path.dirname(os.path.realpath(__file__))


# img_height = 512 # Height of the model input images
# img_width = 512# Width of the model input images
# img_channels = 3 # Number of color channels of the model input images
# mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
# swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
# n_classes = 43 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
# scales_traffic_sign = [0.04, 0.112, 0.184, 0.256, 0.328, 0.4, 0.472]
# # scales = scales_pascal
# scales = scales_traffic_sign
# aspect_ratios = [[1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                  [1.0, 2.0, 0.5],
#                  [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters

 
# two_boxes_for_ar1 = True
# steps = None # The space between two adjacent anchor box center points for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
# clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
# variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
# normalize_coords = True
# batch_size = 8

# # 1: Build the Keras model.

# K.clear_session() # Clear previous models from memory.

# model = SSD(input_shape=(img_height, img_width, img_channels),
#                 num_classes=n_classes,
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


# print('model.summary: ', model.summary())
 
# # weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_last_epoch-92_loss-4.6911_val_loss-7.8811.h5') #best weight

# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_last_epoch-92_loss-4.6911_val_loss-7.8811.h5') #best weight

# # weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-125_loss-4.6402_val_loss-7.7933.h5')
# # weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-105_loss-4.6425_val_loss-7.8029.h5')
# # weight_path = os.path.join(fileDir, 'training_ssd416_mobilenet_epoch-41_loss-6.0904_val_loss1-0.3212.h5')
# model.load_weights(weight_path) 

# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# model.compile(optimizer=sgd, loss=ssd_loss.compute_loss, metrics=['accuracy'])



from models.mobilenetv2ssd512_last import SSD
import cv2
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
fileDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.join(fileDir, '..')


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

weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_last_epoch-92_loss-4.6911_val_loss-7.8811.h5')
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-95_loss-4.7612_val_loss-7.8221.h5')
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-95_loss-4.7612_val_loss-7.8221.h5')
model.load_weights(weight_path) 


# do thoi gian model chay tren mot hinh anh (FPS)
# start_time = time.time()
# img_paths = ['datasets/full/00001.jpg', 'datasets/full/00002.jpg', 'datasets/full/00003.jpg']
# img_paths = ['datasets/full/00001.jpg']
image_dir = '/home/ubuntu/Documents/datn/ssd_keras/datasets/full/00218.jpg'
# img_paths = [image_dir, '/home/ubuntu/Documents/datn/ssd_keras/datasets/full/00022.jpg']
img_paths = ['/home/ubuntu/Documents/datn/ssd_keras/datasets/full/00218.jpg',  
            '/home/ubuntu/Documents/datn/ssd_keras/datasets/full/00090.jpg',  
            '/home/ubuntu/Documents/datn/ssd_keras/datasets/full/00024.jpg',  
            '/home/ubuntu/Documents/datn/ssd_keras/datasets/full/00267.jpg']
for image_dir in img_paths: 
    tt_arr = []
    IMG = cv2.imread(image_dir)
    for tt in range(1): 
        # print('--------------------------tt---------------------------: ', tt)
        # start_time = time.time()
        batch_X = []
        # for filename in img_paths:
        with Image.open(image_dir) as image:
            batch_X.append(np.array(image, dtype=np.uint8))


        gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}

        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height,width=img_width, labels_format=gt_format)
        transformations = [convert_to_3_channels, resize]

        input_images = []
        orig_images = []
        batch = []

        # chuyen doi ve input cua model
        for i in range(len(batch_X)): 
            input_image = []
            orig_image = []
            h, w, _ = batch_X[i].shape
            h_o, w_o = h, w
            H = h // 2
            W = w // 2
            orig_img = batch_X[i].copy()
            for transform in transformations:
                batch_X[i] = transform(batch_X[i])

            # # append anh goc
            # # input_images.append((w, h, orig_img, batch_X[i], w_o, h_o))
            # input_image.append(batch_X[i])
            # orig_image.append((orig_img, w_o, h_o, w, h))

            # splitting images into tiles
            tiles = [(orig_img[x:x+H,y:y+W], y, x) for x in range(0,h,H) for y in range(0,w,W)] 
            for img, x, y in tiles:  
                h, w, _ = img.shape 
                o_img = img.copy()
                for transform in transformations:
                    img = transform(img)  
                # input_images.append((x, y, o_img, img, w_o, h_o)) 
                input_image.append(img)
                orig_image.append((o_img, w_o, h_o, x, y))

            batch.append(input_image)
            orig_images.append(orig_image)

        batch_X = np.array(batch_X) 
        # input_images = np.array(input_images) 

        print('-----------batch X------------: ', batch_X.shape)
        # print('-----------input images-----------: ', input_images.shape)
        
        # colors = plt.cm.hsv(np.linspace(0, 1, 44)).tolist()
        # classes = ['background'] + list(range(43)) 
        y_pred = []
        for b in range(len(batch)):
            ba = batch[b]
            ba = np.array(ba)
            pred= model.predict(ba)
            pred = decode_detections(pred, 
                                    confidence_thresh=0.45,
                                    iou_threshold=0.45,
                                    top_k=100,
                                    input_coords='centroids',
                                    normalize_coords=True,
                                    img_height=img_height,
                                    img_width=img_width)
            y_pred_filtered = []
            orig_image = orig_images[b] 
            current_axis = plt.gca()

            # xet tung tiles image (bao gom ca image goc)
            for i in range(len(pred)): 
                o, w_o, h_o, x, y = orig_image[i] 
                # xet moi bounding box cua tung tiles
                for j in range(pred[i].shape[0]): 
                    # if pred[i][j].all() != 0:
                    # tinh lai toa do bounding box doi voi anh goc 
                    class_id, conf, xmin, ymin, xmax, ymax = pred[i][j] 
                    xmin = xmin * o.shape[1] / img_width  
                    ymin = ymin * o.shape[0] / img_height
                    xmax = xmax * o.shape[1] / img_width 
                    ymax = ymax * o.shape[0] / img_height
                    if x != 0 and x!=w_o:
                        xmin += x
                        xmax += x
                    if y != 0 and y!=h_o:
                        ymin += y
                        ymax += y  
                    xmax = int(xmax)
                    xmin = int(xmin)
                    ymax = int(ymax)
                    ymin = int(ymin)
                    y_pred_filtered.append([class_id, conf, xmin, ymin, xmax, ymax]) 
                    cv2.rectangle(IMG,(xmin, ymin), (xmax, ymax),(0,255,255),2)
                    cv2.putText(IMG, str(int(class_id)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0),2) 

            
            y_pred_filtered = np.array(y_pred_filtered) 
            print('------------------y_predddddddddddddddd--------------------: ', y_pred_filtered)
        
            # y_pred_filtered = sorted(y_pred_filtered, key=operator.itemgetter(1)) 
            # y_pred_filtered = dict((x[0], x) for x in y_pred_filtered).values()
            # y_pred_filtered = list(y_pred_filtered) 
            # y_pred_filtered = np.array(y_pred_filtered) 
            # print('-----------y_pred_filtered-------------: ', y_pred_filtered)
            y_pred.append(y_pred_filtered)
            cv2.imshow("image1",IMG)
            cv2.waitKey(0)
            
        print('-----y_pred-----: ', y_pred[0].shape)

    #     # e = time.time() - start_time
    #     tt_arr.append(time.time() - start_time)
    #     # print('done in : ', e)
    #     # print('FPS: ', 1 / e)
    # print(tt_arr)
    # print('done in: ', sum(tt_arr[1:]) / len(tt_arr[1:]))
        # yy_pred = []

        # y_pred_converted = []
        # y_pred = model.predict(input_images)
        # y_pred_filtered = []
        # for i in range(len(y_pred)):
        #     y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])

        # print(y_pred_filtered)
        # # predict batch_X (orig_image + tiles)
        # for x, y, o, i, w_o, h_o in input_images:   
        #     i = np.array([i])
        #     y_pred = model.predict(i)  
        
        #     confidence_threshold = 0.5 

        #     y_pred_filtered = []
        #     for i in range(len(y_pred)):
        #         y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0]) 

        #     y_pred = y_pred_filtered 

        #     # np.set_printoptions(precision=2, suppress=True, linewidth=90)
        #     # print("Predicted boxes:\n")
        #     # print('   class   conf xmin   ymin   xmax   ymax')
        #     # print(y_pred_thresh[0]) 
        #     # tim lai bounding box cua object tren anh goc
        #     y_pred = np.array(y_pred)
        #     print('shape y_pred: ', y_pred.shape)
        #     for box in y_pred[0]: 
        #         class_id = box[0]
        #         conf = box[1]
        #         xmin = box[2] * o.shape[1] / img_width   + (w_o - x)
        #         ymin = box[3] * o.shape[0] / img_height  + (h_o - y)
        #         xmax = box[4] * o.shape[1] / img_width   + (w_o - x)
        #         ymax = box[5] * o.shape[0] / img_height  + (h_o - y)
        #         y_pred_converted.append([class_id, conf, xmin, ymin, xmax, ymax])
            
        # y_pred_converted = np.array(y_pred_converted)
        # print('--------------------y pred converted-----------------------: ', y_pred_converted[0][0])
        # print('--------------------y pred---------------------------: ', y_pred[0][0])
        #     # plt.show()
