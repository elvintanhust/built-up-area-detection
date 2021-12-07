# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import concatenate
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import maximum
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras.utils import conv_utils
from keras.utils.data_utils import get_file

kernel_init=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
bias_init='zeros'
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


class PanNet():
    def conv_block(self,inputs):
        x = conv2d_bn(inputs, 32, 3, 3,name = 'pan_conv1')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 32 * 32 * 32

        x = conv2d_bn(x, 64, 3, 3,name = 'pan_conv2')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 16 * 16 * 64
        return x

    def inception_block(self,inputs):
        x = inputs  # 16 * 16 * 64
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        branch1x1 = conv2d_bn(x, 64, 1, 1,name = 'pan_m1b1_conv1')

        branch5x5 = conv2d_bn(x, 48, 1, 1,name = 'pan_m1b2_conv1')
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5,name = 'pan_m1b2_conv2')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1,name = 'pan_m1b3_conv1')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,name = 'pan_m1b3_conv2')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,name = 'pan_m1b3_conv3')

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1,name = 'pan_m1b4_conv1')
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='pan_mixed1')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 8 * 8 * 256
        # mixed 2: 8 * 8 * 256
        branch1x1 = conv2d_bn(x, 192, 1, 1,name = 'pan_m2b1_conv1')

        branch7x7 = conv2d_bn(x, 128, 1, 1,name = 'pan_m2b2_conv1')
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7,name = 'pan_m2b2_conv2')
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1,name = 'pan_m2b2_conv3')

        branch7x7dbl = conv2d_bn(x, 128, 1, 1,name = 'pan_m2b3_conv1')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1,name = 'pan_m2b3_conv2')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7,name = 'pan_m2b3_conv3')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1,name = 'pan_m2b3_conv4')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7,name = 'pan_m2b3_conv5')

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1,name = 'pan_m2b4_conv1')
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='pan_mixed2')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 4 * 4 * 768
        # mixed 3: 4 * 4 * 768
        branch1x1 = conv2d_bn(x, 320, 1, 1,name = 'pan_m3b1_conv1')

        branch3x3 = conv2d_bn(x, 384, 1, 1,name = 'pan_m3b2_conv1')
        branch3x3 = conv2d_bn(branch3x3, 384, 1, 3,name = 'pan_m3b2_conv2')
        branch3x3 = conv2d_bn(branch3x3, 384, 3, 1,name = 'pan_m3b2_conv3')

        branch3x3dbl = conv2d_bn(x, 448, 1, 1,name = 'pan_m3b3_conv1')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3,name = 'pan_m3b3_conv2')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 1, 3,name = 'pan_m3b3_conv3')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 1,name = 'pan_m3b3_conv4')


        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1,name = 'pan_m3b4_conv1')
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='pan_mixed3')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 2 * 2 * 1280
        return x

    def fc_and_classify(self,inputs):
        x = inputs
        x = Flatten()(x)
        x = Dense(512,name = 'pan_fc1')(x)
        x = Activation('relu')(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)

        return x

    def __init__(self, size=(64,64,3)):
        self.create(size)

    def create(self, size):
        inputs = Input(shape=size)
        x = self.conv_block(inputs)
        x = self.inception_block(x)
        x = self.fc_and_classify(x)
        self.model = Model(inputs=inputs, outputs=x)

class MulNet():
    def conv_block(self,inputs):
        x = conv2d_bn(inputs, 32, 3, 3,name = 'mul_conv1')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = conv2d_bn(x, 64, 3, 3, name = 'mul_conv2')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 4 * 4 * 64
        return x

    def inception_block(self,inputs):
        # mixed 1, 1, 2: 35 x 35 x 256
        x = inputs
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        branch1x1 = conv2d_bn(x, 64, 1, 1,name = 'mul_m1b1_conv1')

        branch5x5 = conv2d_bn(x, 48, 1, 1,name = 'mul_m1b2_conv1')
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5,name = 'mul_m1b2_conv2')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1,name = 'mul_m1b3_conv1')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,name = 'mul_m1b3_conv2')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,name = 'mul_m1b3_conv3')

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1,name = 'mul_m1b4_conv1')
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mul_mixed1')
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 2 * 2 * 64
        return x

    def fc_and_classify(self,inputs):
        x = inputs
        x = Flatten()(x)
        x = Dense(512,name = 'mul_fc1')(x)
        x = Activation('relu')(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)
        return x

    def __init__(self, size=(16,16,3)):
        self.create(size)

    def create(self, size):
        inputs = Input(shape=size)
        x = self.conv_block(inputs)
        x = self.inception_block(x)
        x = self.fc_and_classify(x)
        self.model = Model(inputs=inputs, outputs=x)

class DoubleStreamNet():
    def conv_block(self,pan_inputs,mul_inputs):
        pan = conv2d_bn(pan_inputs, 32, 3, 3,name = 'pan_conv1')
        pan = MaxPooling2D((2, 2), strides=(2, 2))(pan)

        pan = conv2d_bn(pan, 64, 3, 3,name = 'pan_conv2')
        pan = MaxPooling2D((2, 2), strides=(2, 2))(pan)

        mul = conv2d_bn(mul_inputs, 32, 3, 3, name='mul_conv1')
        mul = MaxPooling2D((2, 2), strides=(2, 2))(mul)

        mul = conv2d_bn(mul, 64, 3, 3, name='mul_conv2')
        mul = MaxPooling2D((2, 2), strides=(2, 2))(mul)
        return pan,mul

    def inception_block(self,pan_inputs,mul_inputs):
        # mixed 1, 1, 2: 35 x 35 x 256
        pan = pan_inputs
        mul = mul_inputs
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        pan_branch1x1 = conv2d_bn(pan, 64, 1, 1,name = 'pan_m1b1_conv1')

        pan_branch5x5 = conv2d_bn(pan, 48, 1, 1,name = 'pan_m1b2_conv1')
        pan_branch5x5 = conv2d_bn(pan_branch5x5, 64, 5, 5,name = 'pan_m1b2_conv2')

        pan_branch3x3dbl = conv2d_bn(pan, 64, 1, 1,name = 'pan_m1b3_conv1')
        pan_branch3x3dbl = conv2d_bn(pan_branch3x3dbl, 96, 3, 3,name = 'pan_m1b3_conv2')
        pan_branch3x3dbl = conv2d_bn(pan_branch3x3dbl, 96, 3, 3,name = 'pan_m1b3_conv3')

        pan_branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pan)
        pan_branch_pool = conv2d_bn(pan_branch_pool, 32, 1, 1,name = 'pan_m1b4_conv1')
        pan = concatenate(
            [pan_branch1x1, pan_branch5x5, pan_branch3x3dbl, pan_branch_pool],
            axis=channel_axis,
            name='pan_mixed1')
        pan = MaxPooling2D((2, 2), strides=(2, 2))(pan)
        # mixed 2: 17 x 17 x 768
        pan_branch1x1 = conv2d_bn(pan, 192, 1, 1,name = 'pan_m2b1_conv1')

        pan_branch7x7 = conv2d_bn(pan, 128, 1, 1,name = 'pan_m2b2_conv1')
        pan_branch7x7 = conv2d_bn(pan_branch7x7, 128, 1, 7,name = 'pan_m2b2_conv2')
        pan_branch7x7 = conv2d_bn(pan_branch7x7, 192, 7, 1,name = 'pan_m2b2_conv3')

        pan_branch7x7dbl = conv2d_bn(pan, 128, 1, 1,name = 'pan_m2b3_conv1')
        pan_branch7x7dbl = conv2d_bn(pan_branch7x7dbl, 128, 7, 1,name = 'pan_m2b3_conv2')
        pan_branch7x7dbl = conv2d_bn(pan_branch7x7dbl, 128, 1, 7,name = 'pan_m2b3_conv3')
        pan_branch7x7dbl = conv2d_bn(pan_branch7x7dbl, 128, 7, 1,name = 'pan_m2b3_conv4')
        pan_branch7x7dbl = conv2d_bn(pan_branch7x7dbl, 192, 1, 7,name = 'pan_m2b3_conv5')

        pan_branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pan)
        pan_branch_pool = conv2d_bn(pan_branch_pool, 192, 1, 1,name = 'pan_m2b4_conv1')
        pan = concatenate(
            [pan_branch1x1, pan_branch7x7, pan_branch7x7dbl, pan_branch_pool],
            axis=channel_axis,
            name='pan_mixed2')
        pan = MaxPooling2D((2, 2), strides=(2, 2))(pan)
        # mixed 3: 8 x 8 x 2048
        pan_branch1x1 = conv2d_bn(pan, 320, 1, 1,name = 'pan_m3b1_conv1')

        pan_branch3x3 = conv2d_bn(pan, 384, 1, 1,name = 'pan_m3b2_conv1')
        pan_branch3x3 = conv2d_bn(pan_branch3x3, 384, 1, 3,name = 'pan_m3b2_conv2')
        pan_branch3x3 = conv2d_bn(pan_branch3x3, 384, 3, 1,name = 'pan_m3b2_conv3')

        pan_branch3x3dbl = conv2d_bn(pan, 448, 1, 1,name = 'pan_m3b3_conv1')
        pan_branch3x3dbl = conv2d_bn(pan_branch3x3dbl, 384, 3, 3,name = 'pan_m3b3_conv2')
        pan_branch3x3dbl = conv2d_bn(pan_branch3x3dbl, 384, 1, 3,name = 'pan_m3b3_conv3')
        pan_branch3x3dbl = conv2d_bn(pan_branch3x3dbl, 384, 3, 1,name = 'pan_m3b3_conv4')

        pan_branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(pan)
        pan_branch_pool = conv2d_bn(pan_branch_pool, 192, 1, 1,name = 'pan_m3b4_conv1')
        pan = concatenate(
            [pan_branch1x1, pan_branch3x3, pan_branch3x3dbl, pan_branch_pool],
            axis=channel_axis,
            name='pan_mixed3')
        pan = MaxPooling2D((2, 2), strides=(2, 2))(pan)
        ##--------------------------------------------------------------------------------------------------------------
        mul_branch1x1 = conv2d_bn(mul, 64, 1, 1, name='mul_m1b1_conv1')

        mul_branch5x5 = conv2d_bn(mul, 48, 1, 1, name='mul_m1b2_conv1')
        mul_branch5x5 = conv2d_bn(mul_branch5x5, 64, 5, 5, name='mul_m1b2_conv2')

        mul_branch3x3dbl = conv2d_bn(mul, 64, 1, 1, name='mul_m1b3_conv1')
        mul_branch3x3dbl = conv2d_bn(mul_branch3x3dbl, 96, 3, 3, name='mul_m1b3_conv2')
        mul_branch3x3dbl = conv2d_bn(mul_branch3x3dbl, 96, 3, 3, name='mul_m1b3_conv3')

        mul_branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(mul)
        mul_branch_pool = conv2d_bn(mul_branch_pool, 32, 1, 1, name='mul_m1b4_conv1')
        mul = concatenate(
            [mul_branch1x1, mul_branch5x5, mul_branch3x3dbl, mul_branch_pool],
            axis=channel_axis,
            name='mul_mixed1')
        mul = MaxPooling2D((2, 2), strides=(2, 2))(mul)
        return pan,mul

    def fc_and_classify(self,pan_inputs,mul_inputs):
        pan = Flatten()(pan_inputs)
        pan = Dense(512,name = 'pan_fc1')(pan)
        pan = Activation('relu')(pan)

        mul = Flatten()(mul_inputs)
        mul = Dense(512, name='mul_fc1')(mul)
        mul = Activation('relu')(mul)

        out = maximum([pan,mul],name = 'MaxOut')
        out = Dense(2)(out)
        out = Activation('softmax')(out)

        return out

    def __init__(self, pan_size=(64,64,1),mul_size=(16,16,3)):
        self.create(pan_size,mul_size)

    def create(self, pan_size,mul_size):
        pan_inputs = Input(shape=pan_size)
        mul_inputs = Input(shape=mul_size)
        pan, mul = self.conv_block(pan_inputs,mul_inputs)
        pan, mul = self.inception_block(pan, mul)
        out = self.fc_and_classify(pan, mul)
        self.model = Model(inputs=[pan_inputs, mul_inputs], outputs=out)