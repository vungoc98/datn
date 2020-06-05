"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Conv2D,DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Flatten,Add
from keras.layers import Input
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers import Reshape, Lambda, ZeroPadding2D, Concatenate
from keras.models import Model
# from ssd_layers import PriorBox
from keras.applications import MobileNetV2
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
import numpy as np
from keras.regularizers import l2
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

def relu6(x):
    return K.relu(x, max_value=6)

def LiteConv(x,i,filter_num):
    x = Conv2D(filter_num//2, (1, 1), padding='same', use_bias=False, name=str(i)+'_pwconv1')(x)
    x = BatchNormalization(momentum=0.99,name=str(i)+'_pwconv1_bn')(x)
    x = Activation('relu', name=str(i)+'_pwconv1_act')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name=str(i)+'_dwconv2')(x)
    x = BatchNormalization(momentum=0.99,name=str(i) + '_sepconv2_bn')(x)
    x = Activation('relu', name=str(i) + '_sepconv2_act')(x)
    net = Conv2D(filter_num, (1, 1), padding='same', use_bias=False, name=str(i) + '_pwconv3')(x)
    x = BatchNormalization(momentum=0.99,name=str(i) + '_pwconv3_bn')(net)
    x = Activation('relu', name=str(i) + '_pwconv3_act')(x)
    print(x.shape)
    return x,net

def Conv(x,filter_num):
    net = Conv2D(filter_num,kernel_size=1,strides=(2,2),use_bias=False,name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(net)
    x = Activation(relu6, name='Conv_1_relu')(x)
    print(x.shape)
    return x,net


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Expand
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',use_bias=False, activation=None,name='mobl%d_conv_expand' % block_id)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='bn%d_conv_bn_expand' %block_id)(x)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,use_bias=False, padding='same',name='mobl%d_conv_depthwise' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='bn%d_conv_depthwise' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    # Project
    x = Conv2D(pointwise_filters,kernel_size=1, padding='same', use_bias=False, activation=None,name='mobl%d_conv_project' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='bn%d_conv_bn_project' % block_id)(x)
    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])
    return x

def _isb4conv13(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Expand
    conv = Conv2D(expansion * in_channels, kernel_size=1, padding='same',use_bias=False, activation=None,name='mobl%d_conv_expand' % block_id)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='bn%d_conv_bn_expand' %block_id)(conv)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,use_bias=False, padding='same',name='mobl%d_conv_depthwise' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='bn%d_conv_depthwise' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    # Project
    x = Conv2D(pointwise_filters,kernel_size=1, padding='same', use_bias=False, activation=None,name='mobl%d_conv_project' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='bn%d_conv_bn_project' % block_id)(x)
    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])
    return x,conv



def prediction(x,i,num_priors,min_s,max_s,aspect,num_classes,img_size):
    a=Conv2D(num_priors*4,(3,3),padding='same',name=str(i)+'_mbox_loc')(x)
    mbox_loc_flat=Flatten(name=str(i)+'_mbox_loc_flat')(a)
    b=Conv2D(num_priors*num_classes,(3,3),padding='same',name=str(i)+'_mbox_conf')(x)
    mbox_conf_flat=Flatten(name=str(i)+'_mbox_conf_flat')(b)
    mbox_priorbox=PriorBox(img_size,min_size=min_s,max_size=max_s,aspect_ratios=aspect,variances=[0.1,0.1,0.2,0.2],name=str(i)+'_mbox_priorbox')(x)
    return mbox_loc_flat,mbox_conf_flat,mbox_priorbox


def SSD(input_shape, num_classes,
        mode='training',
        l2_regularization=0.0005,
        min_scale=None,
        max_scale=None,
        scales=None,
        aspect_ratios_global=None,
        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                    [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                    [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                    [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                    [1.0, 2.0, 0.5],
                                    [1.0, 2.0, 0.5]],
        two_boxes_for_ar1=True,
        steps=[8, 16, 32, 64, 100, 300],
        offsets=None,
        clip_boxes=False,
        variances=[0.1, 0.1, 0.2, 0.2],
        coords='centroids',
        normalize_coords=True,
        subtract_mean=[123, 117, 104],
        divide_by_stddev=None,
        swap_channels=[2, 1, 0],
        confidence_thresh=0.01,
        iou_threshold=0.45,
        top_k=200,
        nms_max_output_size=400,
        return_predictor_sizes=False):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    num_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = input_shape[0], input_shape[1], input_shape[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    alpha=1.0
    img_size=(input_shape[1],input_shape[0])
    input_shape=(input_shape[1],input_shape[0],3)
    mobilenetv2_input_shape=(224,224,3)

    Input0 = Input(input_shape)

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(Input0)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)
 

    mobilenetv2=MobileNetV2(input_shape=mobilenetv2_input_shape,include_top=False,weights='imagenet')
    FeatureExtractor=Model(inputs=mobilenetv2.input, outputs=mobilenetv2.get_layer('block_12_add').output)
    #get_3rd_layer_output = K.function([mobilenetv2.layers[114].input, K.learning_phase()],
    #                                  [mobilenetv2.layers[147].output])

    x= FeatureExtractor(x1)
    x,pwconv3 = _isb4conv13(x, filters=160, alpha=alpha, stride=1,expansion=6, block_id=13)
    #x=get_3rd_layer_output([x,1])[0]
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,expansion=6, block_id=15)
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,expansion=6, block_id=16)
    x, pwconv4 = Conv(x, 1280)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(pwconv4)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    print('-----------------conv9-1---------------------: ', conv9_1.shape)
    conv9_2 = Conv2D(256, (1, 1), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(pwconv3)
 
    ### Build the convolutional predictor layers on top of the base network

    # We precidt `num_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * num_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * num_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(pwconv3)
    fc7_mbox_conf = Conv2D(n_boxes[1] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(pwconv4)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * num_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(pwconv3)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(pwconv4)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, num_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, num_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, num_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, num_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, num_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, num_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, num_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (num_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, num_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, num_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=Input0, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=Input0, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=Input0, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))
    # x, pwconv5 = LiteConv(x, 5, 512)
    # x, pwconv6 = LiteConv(x, 6, 256)
    # x, pwconv7 = LiteConv(x, 7, 128)
    # x, pwconv8 = LiteConv(x, 8, 128)

    # pwconv3_mbox_loc_flat, pwconv3_mbox_conf_flat, pwconv3_mbox_priorbox = prediction(pwconv3, 3, 3, 60.0 ,None ,[2],num_classes, img_size)
    # pwconv4_mbox_loc_flat, pwconv4_mbox_conf_flat, pwconv4_mbox_priorbox = prediction(pwconv4, 4, 6, 105.0,150.0,[2, 3], num_classes, img_size)
    # pwconv5_mbox_loc_flat, pwconv5_mbox_conf_flat, pwconv5_mbox_priorbox = prediction(pwconv5, 5, 6, 150.0,195.0,[2, 3], num_classes, img_size)
    # pwconv6_mbox_loc_flat, pwconv6_mbox_conf_flat, pwconv6_mbox_priorbox = prediction(pwconv6, 6, 6, 195.0,240.0,[2, 3], num_classes, img_size)
    # pwconv7_mbox_loc_flat, pwconv7_mbox_conf_flat, pwconv7_mbox_priorbox = prediction(pwconv7, 7, 6, 240.0,285.0,[2, 3], num_classes, img_size)
    # pwconv8_mbox_loc_flat, pwconv8_mbox_conf_flat, pwconv8_mbox_priorbox = prediction(pwconv8, 8, 6, 285.0,300.0,[2, 3],num_classes, img_size)


    # # Gather all predictions
    # mbox_loc = concatenate(
    #     [pwconv3_mbox_loc_flat, pwconv4_mbox_loc_flat, pwconv5_mbox_loc_flat, pwconv6_mbox_loc_flat,
    #      pwconv7_mbox_loc_flat, pwconv8_mbox_loc_flat], axis=1, name='mbox_loc')
    # mbox_conf = concatenate(
    #     [pwconv3_mbox_conf_flat, pwconv4_mbox_conf_flat, pwconv5_mbox_conf_flat, pwconv6_mbox_conf_flat,
    #      pwconv7_mbox_conf_flat, pwconv8_mbox_conf_flat], axis=1, name='mbox_conf')
    # mbox_priorbox = concatenate(
    #     [pwconv3_mbox_priorbox, pwconv4_mbox_priorbox, pwconv5_mbox_priorbox, pwconv6_mbox_priorbox,
    #      pwconv7_mbox_priorbox, pwconv8_mbox_priorbox], axis=1, name='mbox_priorbox')
    # if hasattr(mbox_loc, '_keras_shape'):
    #     num_boxes = mbox_loc._keras_shape[-1] // 4
    # elif hasattr(mbox_loc, 'int_shape'):
    #     num_boxes = K.int_shape(mbox_loc)[-1] // 4
    # mbox_loc = Reshape((num_boxes, 4),name='mbox_loc_final')(mbox_loc)
    # mbox_conf = Reshape((num_boxes, num_classes),name='mbox_conf_logits')(mbox_conf)
    # mbox_conf = Activation('softmax',name='mbox_conf_final')(mbox_conf)
    # predictions = concatenate([mbox_loc,mbox_conf,mbox_priorbox],axis=2,name='predictions')
    # model = Model(inputs=Input0,  outputs=predictions)
    return model