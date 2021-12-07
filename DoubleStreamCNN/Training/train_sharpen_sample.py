# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from NET.network import PanNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))

def printParams(model,layer):
    param = model.layers[layer].get_weights()
    kernel_param = param[0]  # get the kernel
    print(kernel_param[:,:,0,0])

def training(batch_size=100, learn_rate=0.0003):
    panheight = 64
    panwidth = 64
    mulheight = 16
    mulwidth = 16
    # -------------------------------------------------------------------------------------------------------------------

    MODEL = PanNet()

    # initiate RMSprop optimizer
    # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)

    opt = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])


    # ------------------------------------------------------------------------------------------------------------------
    saver_dir = r'F:\硕士研究生学习\建成区提取\Built-Up Area Detection Contrast Algorithms\Supervised Approaches\DoubleStreamCNN(IEEE_J_Stars)\models\Sharpen\model.h5'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.05,
        # zoom_range=0.05,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'F:/数据备份/Remote Sensing/GF2/fuseImage',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        'F:/数据备份/Remote Sensing/GF2/fuseImageTest',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical')

    MODEL.model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=200)
    MODEL.model.save_weights(saver_dir)
    return True


if __name__ == "__main__":

    training( batch_size=32, learn_rate=0.0001)

