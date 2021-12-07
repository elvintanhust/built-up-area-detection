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
from tensorflow.contrib.keras.python.keras.layers import Flatten
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import Reshape
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file

kernel_init=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
bias_init='zeros'

class MobileNet():
    def mobile_block(self, filter_1, filter_2):
        model = self.model
        model.add(SeparableConv2D(filter_1,kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_1,kernel_size=(1,1),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(filter_2, kernel_size=(3,3), strides=(2,2),padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_2 * 2,kernel_size=(1,1),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def final_conv_block(self):
        model = self.model
        model.add(SeparableConv2D(512,kernel_size=(3,3), strides=(2,2),padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(1024,kernel_size=(1,1),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(1024,kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(1024,kernel_size=(1,1),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def separable_filters(self):
        model = self.model
        for i in range(1):
            model.add(SeparableConv2D(512,kernel_size=(3,3), strides=(1,1),padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(512,kernel_size=(1,1),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

    def pool_and_classify(self):
        model = self.model
        model.add(AveragePooling2D(pool_size=(2,2),strides=(1,1)))
        model.add(Flatten())
        model.add(Dense(2)) #按照自己的分类数目进行修改Dense()
        model.add(Activation('softmax'))

    def __init__(self, size=(128,128,1)):
        self.create(size)

    def create(self, size):
        self.model = Sequential()
        self.model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2), padding='same', input_shape=size))
        self.mobile_block(32,64)
        self.mobile_block(128,128)  # 32 * 32 * 128
        self.mobile_block(256,256)  # 16 * 16 * 512
        self.separable_filters()  # 8 * 8 * 512
        self.final_conv_block()  # 4 * 4 * 1024
        self.pool_and_classify()