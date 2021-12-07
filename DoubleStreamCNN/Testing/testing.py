# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

from NET.network import PanNet, MulNet, DoubleStreamNet

from Function.function import getDataByList
from Function.TFRecord2image import next_city_batch
from Function.TFRecord2image import next_noncity_batch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def testing(modelName='Mobile', fileset='test', batch_size=100,
            saver_dir=r'D:\MyProject\Tensorflow\workspace\models\Inception\2017-11-10_23-14-00'):
    height, width = 128, 128
    data = getDataByList(model=fileset, batch_size=100)

    if modelName == 'Mobile':
        MODEL = MobileNet()
    elif modelName == 'Alexnet':
        MODEL = AlexNet()
    elif modelName == 'Inception':
        MODEL = Inception()
    elif modelName == 'Tinydark':
        MODEL = TinyDark()
    model_name = '%s_model.h5' % modelName
    MODEL.model.load_weights(os.path.join(saver_dir, model_name))

    # initiate RMSprop optimizer
    # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    n1 = 0
    acc1 = 0
    loss1 = 0
    data_time = 0
    start_t = time.time()
    while 1:
        temp_time = time.time()
        x, end = data.get_city_batch(singleloop=True, preprocess=False)
        data_time += time.time() - temp_time
        if end:
            break
        score = MODEL.model.test_on_batch(x, data.test_label_c, sample_weight=None)
        loss1 += score[0]
        acc1 += score[1]
        n1 += 1
        print('第%d批数据  \r\n' % n1)
    print("Loss: %.4f  ACC: %.4f\n" % (loss1 / n1, acc1 / n1))
    n2 = 0
    acc2 = 0
    loss2 = 0
    while 1:
        temp_time = time.time()
        x, end = data.get_noncity_batch(singleloop=True, preprocess=False)
        data_time += time.time() - temp_time
        if end:
            break
        score = MODEL.model.test_on_batch(x, data.test_label_nc, sample_weight=None)
        loss2 += score[0]
        acc2 += score[1]
        n2 += 1
        print('第%d批数据  \r\n' % n2)
    print("Loss: %.4f  ACC: %.4f\n" % (loss2 / n2, acc2 / n2))
    times = time.time() - start_t
    print("times: %d  imnum_per_sec:%d\n" % (times, (data.cityNum + data.noncityNum) / times))
    print(data_time / times)
    print("Total Loss: %.4f  Total ACC: %.4f\n" % ((loss1 + loss2) / (n1 + n2), (acc1 + acc2) / (n1 + n2)))


def testing_from_tfrecord(modelName='PanNet', batch_size=100,saver_dir=None):
    panheight = 64
    panwidth = 64
    mulheight = 16
    mulwidth = 16

    with tf.Session() as sess:
        pan_city_image_batch, mul_city_image_batch, city_label_batch = next_city_batch(model='test', num_epochs=1,
                                                                                       batch_size=batch_size,
                                                                                       pan_height=panheight,
                                                                                       pan_width=panwidth,
                                                                                       mul_height=mulheight,
                                                                                       mul_width=mulwidth, n_classes=2,
                                                                                       isProcess=False)
        pan_noncity_image_batch, mul_noncity_image_batch, noncity_label_batch = next_noncity_batch(model='test',
                                                                                                   num_epochs=1,
                                                                                                   batch_size=batch_size,
                                                                                                   pan_height=panheight,
                                                                                                   pan_width=panwidth,
                                                                                                   mul_height=mulheight,
                                                                                                   mul_width=mulwidth,
                                                                                                   n_classes=2,
                                                                                                   isProcess=False)
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if modelName == 'PanNet':
            MODEL = PanNet()
        elif modelName == 'MulNet':
            MODEL = MulNet()
        elif modelName == 'DoubleStreamNet':
            MODEL = DoubleStreamNet()
        model_name = '10%s_model.h5' % (modelName,)
        MODEL.model.load_weights(os.path.join(saver_dir, model_name))

        # initiate RMSprop optimizer
        # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
        opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Let's train the model using RMSprop
        MODEL.model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
        data_time = 0
        predict_time = 0
        n1 = 0
        acc1 = 0
        loss1 = 0
        while 1:
            try:
                temp_time = time.time()
                c_pan_image_batch, c_mul_image_batch, c_label_batch = sess.run(
                    [pan_city_image_batch, mul_city_image_batch, city_label_batch])
                data_time += time.time() - temp_time
            except tf.errors.OutOfRangeError:
                break
            temp_time = time.time()
            if modelName == 'PanNet':
                score = MODEL.model.test_on_batch(c_pan_image_batch, c_label_batch, sample_weight=None)
            elif modelName == 'MulNet':
                score = MODEL.model.test_on_batch(c_mul_image_batch, c_label_batch, sample_weight=None)
            else:
                score = MODEL.model.test_on_batch([c_pan_image_batch, c_mul_image_batch], c_label_batch,
                                                  sample_weight=None)
            predict_time += time.time() - temp_time
            loss1 += score[0]
            acc1 += score[1]
            n1 += 1
            print('City 第%d批数据\n' % n1)

        n2 = 0
        acc2 = 0
        loss2 = 0
        while 1:
            try:
                temp_time = time.time()
                n_pan_image_batch, n_mul_image_batch, n_label_batch = sess.run(
                    [pan_noncity_image_batch, mul_noncity_image_batch, noncity_label_batch])
                data_time += time.time() - temp_time
            except tf.errors.OutOfRangeError:
                break;
            temp_time = time.time()
            if modelName == 'PanNet':
                score = MODEL.model.test_on_batch(n_pan_image_batch, n_label_batch, sample_weight=None)
            elif modelName == 'MulNet':
                score = MODEL.model.test_on_batch(n_mul_image_batch, n_label_batch, sample_weight=None)
            else:
                score = MODEL.model.test_on_batch([n_pan_image_batch, n_mul_image_batch], n_label_batch,
                                                  sample_weight=None)
            predict_time += time.time() - temp_time
            loss2 += score[0]
            acc2 += score[1]
            n2 += 1
            print('Noncity 第%d批数据\n' % n2)

        print("City Loss: %.4f  City ACC: %.4f\n" % (loss1 / n1, acc1 / n1))
        print("Noncity Loss: %.4f  Noncity ACC: %.4f\n" % (loss2 / n2, acc2 / n2))

        print("Predict times: %d  imnum_per_sec:%d\n" % (predict_time, (n1 + n2) * batch_size / predict_time))
        print("data_time/predict_time = %.4f" % (data_time / predict_time,))
        print("Total Loss: %.4f  Total ACC: %.4f\n" % ((loss1 + loss2) / (n1 + n2), (acc1 + acc2) / (n1 + n2)))
        coord.request_stop()
        coord.join()


if __name__ == "__main__":
    # modelName = ['PanNet'  'MulNet'  'DoubleStreamNet']
    saver_dir = r'D:\MyProject\IEEE_J_Stars1\models\DoubleStreamNet\2018-01-25_21-02-22-CE'
    # testing(modelName='Inception', fileset='test', batch_size=100, saver_dir=saver_dir)
    testing_from_tfrecord(modelName='DoubleStreamNet',batch_size=100, saver_dir=saver_dir)