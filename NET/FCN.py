# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
from tensorflow.contrib.keras.python.keras.layers import Activation
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import Conv2D
from tensorflow.contrib.keras.python.keras.layers import SeparableConv2D
from tensorflow.contrib.keras.python.keras.layers import AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Flatten
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import concatenate
from tensorflow.contrib.keras.python.keras.layers import maximum
from tensorflow.contrib.keras.python.keras.layers import Conv2DTranspose
from tensorflow.contrib.keras.python.keras.layers import Lambda
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import add
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import Reshape
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file

kernel_init=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
bias_init='zeros'

class FCN():
    def mobile_block_branch(self, x, filter_1, filter_2,BN = True):
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3

        x = SeparableConv2D(filter_1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if BN:
            x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Activation('elu')(x)

        x = SeparableConv2D(filter_2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        if BN:
            x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Activation('elu')(x)

        return x

    def dark_block_branch(self, x, filter_1, filter_2,BN = True):
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3

        x = Conv2D(filter_1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)
        if BN:
            x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Activation('elu')(x)

        x = Conv2D(filter_2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)
        if BN:
            x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Activation('elu')(x)

        return x

    def pool_block_branch(self, x, filter):
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(filter, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)
        x = Activation('elu')(x)
        return x

    def conv_block(self, inputs):
        x = inputs
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        # mixed0
        branch1 = self.dark_block_branch(x, 32, 64)
        branch2 = self.mobile_block_branch(x, 32, 64)
        branch3 = self.pool_block_branch(x, 64)
        x = maximum([branch1, branch2, branch3], name='mixed0')

        # mixed1
        branch1 = self.dark_block_branch(x, 64, 96, False)
        branch2 = self.mobile_block_branch(x, 64, 96, False)
        branch3 = self.pool_block_branch(x, 96)
        x = maximum([branch1, branch2, branch3], name='mixed1')

        # # mixed2
        # branch1 = self.dark_block_branch(x, 16, 8, False)
        # branch2 = self.mobile_block_branch(x, 16, 8, False)
        # branch3 = self.pool_block_branch(x, 8)
        # x = maximum([branch1, branch2, branch3], name='mixed2')

        x = SeparableConv2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)
        x = Activation('sigmoid')(x)
        return x



    def __init__(self, size=(60, 60, 256)):
        self.create(size)

    def create(self, size):
        inputs = Input(shape=size)
        x = self.conv_block(inputs)
        self.model = Model(inputs=inputs, outputs=x)


