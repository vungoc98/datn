from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
import os
from matplotlib import pyplot as plt  
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from keras.callbacks import LambdaCallback
import json

# from models.mobilenetv2 import SSD
  
from scipy.misc import imread     
from eval_utils.average_precision_evaluator import Evaluator
# from models.mobilenetv2SSD512 import SSD
model_mode = 'inference'

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
# scales_traffic_sign_split = [0.04, 0.212, 0.384, 0.556, 0.728, 0.9, 1.072]
scales = scales_pascal
# scales = scales_traffic_sign
# scales = scales_traffic_sign_split
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

# weight_path = os.path.join(rootDir, 'inference/training_ssd512_mobilenet-allDATA-newAUG_epoch-96_loss-4.7865_val_loss-7.9207.h5')
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_last_epoch-92_loss-4.6911_val_loss-7.8811.h5')
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-95_loss-4.7612_val_loss-7.8221.h5')
# weight_path = os.path.join(rootDir, 'inference/mobilenetv2/training_mobilenetv2ssd512_class_weight_last_epoch-95_loss-4.7612_val_loss-7.8221.h5')
# weight_path = os.path.join(rootDir, 'inference/m2ssd/training_mobilenetv2ssd512_class_weight_usetraintest_last_epoch-91_loss-5.5499_val_loss-8.2515.h5') #best

## odd scales
# weight_path = os.path.join(rootDir, 'inference/m2ssdoddscales/training_mobilenetv2ssd512_last_odd_scales_last_epoch-108_loss-4.9453_val_loss-7.8760.h5')

## new scales
# weight_path = os.path.join(rootDir, 'inference/m2ssdnewscales/training_mobilenetv2ssd512_last_new_scales_last_epoch-120_loss-4.6292_val_loss-9.0608.h5')
# weight_path = os.path.join(rootDir, 'inference/m2ssdnewscales/training_mobilenetv2ssd512_class_weight_usetraintest_last_epoch-91_loss-5.5499_val_loss-8.2515.h5') #best

## new scales + splitting image
# weight_path = os.path.join(rootDir, 'inference/m2ssdnewscales_splittingimage/training_mobilenetv2ssd512_new_scales_splitting_image_epoch-101_loss-5.1552_val_loss-7.9082.h5')

## odd scales + splitting image
weight_path = os.path.join(rootDir, 'inference/m2ssdoodscales_splittingimage/training_mobilenetv2ssd512_odd_scales_splitting_image_epoch-92_loss-4.3373_val_loss-6.6256.h5')


### new scales + splitting image (tinh lai scales khi su dung splitting image)
# weight_path = os.path.join(rootDir, 'inference/m2ssdnewscales_splittingimage_newratios/training_mobilenetv2ssd512_scales_for_splitting_image_epoch-105_loss-4.7750_val_loss-7.3927.h5')
model.load_weights(weight_path) 


dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
 
classes = ['background'] + list(range(1, 44))
print('classes: ', classes)

# Images 
image_dir = '/home/ubuntu/Documents/datn/ssd_keras/datasets/full'
# val = 'test.csv'
# val = os.path.join(rootDir, 'training/test_new.csv')
test = os.path.join(rootDir, 'training/val_new.csv')
dataset.parse_csv(image_dir, test, input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'])

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.45,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results


for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes: break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0,1,11))
        cells[i, j].set_yticks(np.linspace(0,1,11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)



for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3))) 