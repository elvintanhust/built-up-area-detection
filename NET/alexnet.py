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

kernel_init=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
bias_init='zeros'

class AlexNetCJ():
    def conv_block(self, size):
        model = self.model
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid',kernel_initializer=kernel_init,bias_initializer=bias_init, input_shape=size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2),padding='valid'))

        model.add(Conv2D(256,kernel_size=(5,5),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='valid'))

        model.add(Conv2D(384,kernel_size=(3,3),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    def final_classify(self):
        model = self.model
        model.add(Flatten())
        model.add(Dense(512,kernel_initializer=kernel_init,bias_initializer=bias_init))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(512,kernel_initializer=kernel_init,bias_initializer=bias_init))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(2))  # 按照自己的分类数目进行修改Dense()
        model.add(Activation('softmax'))

    def __init__(self, size=(128,128,1)):
        self.create(size)

    def create(self, size):
        self.model = Sequential()
        self.conv_block(size)
        self.final_classify()

class AlexNet():
    def conv_block(self, size):
        model = self.model
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid',kernel_initializer=kernel_init,bias_initializer=bias_init, input_shape=size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2),padding='valid'))

        model.add(Conv2D(256,kernel_size=(5,5),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='valid'))

        model.add(Conv2D(384,kernel_size=(3,3),strides=(1,1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    def final_classify(self):
        model = self.model
        model.add(Flatten())
        model.add(Dense(1024,kernel_initializer=kernel_init,bias_initializer=bias_init))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1024,kernel_initializer=kernel_init,bias_initializer=bias_init))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(2))  # 按照自己的分类数目进行修改Dense()
        model.add(Activation('softmax'))

    def __init__(self, size=(227,227,1)):
        self.create(size)

    def create(self, size):
        self.model = Sequential()
        self.conv_block(size)
        self.final_classify()