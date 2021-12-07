# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
from Function.function import plot_history

from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.utils import plot_model

from NET.mobilenet import MobileNet
from NET.alexnet import AlexNet
from NET.tinydark import TinyDark
from NET.inception import Inception


def training(modelName = 'Mobile',batch_size = 100,learn_rate =0.0003,epochs = 30,steps_per_epoch = 1000,validation_steps = 500, isprep = False):

    height, width = 128, 128

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # shear_range=0.1,
        # zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'E:/DATA/GF2/panchromatic64/train',
        target_size=(height, width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        'E:/DATA/GF2/panchromatic64/val',
        target_size=(height, width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

    if modelName == 'Mobile':
        MODEL = MobileNet()
    elif modelName == 'Alexnet':
        MODEL = AlexNet()
    elif modelName == 'Inception':
        MODEL = Inception()
    elif modelName == 'Tinydark':
        MODEL = TinyDark()
    # initiate RMSprop optimizer
    # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
    opt = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    history = MODEL.model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=[4.85, 1.0])
    # callbacks= callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto'))
    score = MODEL.model.evaluate_generator(validation_generator, validation_steps, max_q_size=10, workers=1,
                                           pickle_safe=False)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #----------------------------------------------------------------------------------------------------
    times = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    root_dir = 'D:\\MyProject\\Tensorflow\\workspace\\models\\%s' % modelName
    saver_dir = os.path.join(root_dir, times)
    os.mkdir(saver_dir)
    model_name = '%s_model.h5' % modelName
    MODEL.model.save_weights(os.path.join(saver_dir, model_name))

    f = open(os.path.join(saver_dir, 'history.log'),'a+')
    f.write('modelName=%s, batch_size=%d, learn_rate =%f, epochs=%d, steps_per_epoch=%d, validation_steps=%d, isprep = %d\n' % (
        modelName, batch_size, learn_rate, epochs, steps_per_epoch, validation_steps,isprep))
    f.write('\n%-10s%-10s%-10s%-10s\n'%('acc','val_acc','loss','val_loss'))
    for i in range(epochs):
        f.write('%.6f  %.6f  %.6f  %.6f\n' % (history.history['acc'][i],history.history['val_acc'][i],history.history['loss'][i],history.history['val_loss'][i]))
    f.close()
    plot_history(history, saver_dir)



if __name__ == "__main__":
    # modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark']
    training(modelName='Inception', batch_size=64, learn_rate=0.0008, epochs=200, steps_per_epoch=1000,
             validation_steps=500, isprep=False)


    # MODEL = MobileNet()
    # MODEL = AlexNet()
    # MODEL = Inception()
    # MODEL = TinyDark()
    # plot_model(MODEL.model, to_file='E:/Tinydark.png')





# import os
# import cv2
# def getfilelist(file_dir):
#     # 从 TXT文件读取图像路径和标签
#     with open(file_dir) as fid:
#         file = fid.read()
#     content = file.split('\n')
#     return content
#
# Files = getfilelist('D:/MyProject/Tensorflow/workspace/MobileNet/filelist/test_noncity.txt')
# for i in Files:
#     srcdir = 'E:/DATA/GF2/panchromatic64/noncity'
#     dstdir = 'E:/DATA/GF2/panchromatic64/test/noncity'
#     name = os.path.basename(i)
#     im = cv2.imread(os.path.join(srcdir, name))
#     cv2.imwrite(os.path.join(dstdir, name),im)
#     print(i)
