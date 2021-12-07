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
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import Reshape
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
from tensorflow.contrib.keras.python.keras import initializers

kernel_init=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
bias_init='zeros'

class TinyDark():

    def tiny_block(self, filter_1, filter_2, ispooling):
        model = self.model
        model.add(Conv2D(filter_1, kernel_size=(1, 1), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_2, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_1, kernel_size=(1, 1), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(filter_2, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if ispooling:
            model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 2), padding='same'))

    def conv_block(self, size):
        model = self.model
        model.add(Conv2D(16, kernel_size=(3,3),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init, input_shape=size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2),padding='same'))

        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init, input_shape=size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.tiny_block(16, 128, True)
        self.tiny_block(32, 256, True)
        self.tiny_block(64, 512, False)

        model.add(Conv2D(128,kernel_size=(1,1),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 3), strides=(2, 2), padding='same'))


    def final_classify(self):
        model = self.model
        model.add(Flatten())
        model.add(Dense(2))  # 按照自己的分类数目进行修改Dense()
        model.add(Activation('softmax'))

    def __init__(self, size=(128,128,1)):
        self.create(size)

    def create(self, size):
        self.model = Sequential()
        self.conv_block(size)
        self.final_classify()