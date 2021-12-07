# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from LOSS.myloss import NCA_loss
import numpy as np
import tensorflow as tf
from Function.function import show_batch
from Function.function import plot_curv
from Function.function import getDataByList
from Function.TFRecord2image import next_city_batch
from Function.TFRecord2image import next_noncity_batch

from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator

from NET.mobilenet import MobileNet
from NET.alexnet import AlexNet
from NET.tinydark import TinyDark
from NET.inception import Inception
from NET.mynet import MyNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def training(modelName='Mobile', batch_size=100, learn_rate=0.0003, epochs=30, steps_per_epoch=1000,
             validation_steps=500, isprep=False):
    height, width = 128, 128

    trainData = getDataByList(model='train', batch_size=batch_size, height=height, width=width)
    valData = getDataByList(model='val', batch_size=batch_size, height=height, width=width)
    # -------------------------------------------------------------------------------------------------------------------
    if modelName == 'Mobile':
        MODEL = MobileNet()
    elif modelName == 'Alexnet':
        MODEL = AlexNet()
    elif modelName == 'Inception':
        MODEL = Inception()
    elif modelName == 'Tinydark':
        MODEL = TinyDark()
    elif modelName == 'Mynet':
        MODEL = MyNet()

    # initiate RMSprop optimizer
    # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
    opt = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
    # ------------------------------------------------------------------------------------------------------------------

    y = np.zeros([batch_size, 2], float)
    y_c = np.zeros([batch_size, 2], float)
    y_nc = np.zeros([batch_size, 2], float)
    y[0:(batch_size // 2), 0] = 1
    y[(batch_size // 2):batch_size, 1] = 1
    y_c[:, 0] = 1
    y_nc[:, 1] = 1
    train_history = []
    train_history_50batch = []
    val_history = []
    for i in range(1, epochs + 1):
        # 迭代训练一个epoch
        loss, acc = 0, 0
        for j in range(1, steps_per_epoch + 1):
            x, _ = trainData.get_next_batch(singleloop=False, preprocess=isprep)
            # show_batch('x',x,False,8,8,1)
            out = MODEL.model.train_on_batch(x, y, class_weight=None, sample_weight=None)
            if j % 50 == 0:
                train_history_50batch.append(out)
                print('epoch %d batch iter %d ====> loss = %.6f  acc = %.6f' % (i, j, out[0], out[1]))
            # 测试训练集上性能
            if j > steps_per_epoch - 100:
                loss += out[0]
                acc += out[1]
        loss = loss / 100
        acc = acc / 100
        train_history.append([loss, acc])

        # 测试验证集上性能
        val_loss1, val_acc1, val_loss2, val_acc2 = 0, 0, 0, 0
        val_out = []
        for j in range(1, validation_steps + 1):
            city_batch, _ = valData.get_city_batch(singleloop=False, preprocess=False)
            noncity_batch, _ = valData.get_noncity_batch(singleloop=False, preprocess=False)
            out1 = MODEL.model.evaluate(city_batch, y_c, batch_size=batch_size, verbose=0)
            out2 = MODEL.model.evaluate(noncity_batch, y_nc, batch_size=batch_size, verbose=0)
            val_loss1 += out1[0]
            val_acc1 += out1[1]
            val_loss2 += out2[0]
            val_acc2 += out2[1]
        val_out.append(val_loss1)
        val_out.append(val_acc1)
        val_out.append(val_loss2)
        val_out.append(val_acc2)
        val_out = np.array(val_out)
        val_out = val_out / validation_steps
        print('epoch %d ====>training acc = %.6f     city acc = %.6f  noncity acc = %.6f' % (
        i, acc, val_out[1], val_out[3]))
        val_history.append(val_out)

    # -------------------------------------------------------------------------------------------------------------------
    times = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '-CE'
    root_dir = 'D:\\MyProject\\Tensorflow\\workspace\\models\\%s' % modelName
    saver_dir = os.path.join(root_dir, times)
    os.mkdir(saver_dir)
    model_name = '%s_model.h5' % modelName
    MODEL.model.save_weights(os.path.join(saver_dir, model_name))

    f = open(os.path.join(saver_dir, 'history.log'), 'a+')
    f.write('focalLoss + 数据均衡\n')
    f.write(
        'modelName=%s, batch_size=%d, learn_rate =%f, epochs=%d, steps_per_epoch=%d, validation_steps=%d, isprep = %d\n' % (
            modelName, batch_size, learn_rate, epochs, steps_per_epoch, validation_steps, isprep))
    f.write('\n%-10s%-10s%-10s%-10s%-10s%-10s\n' % ('t_loss', 't_acc', 'c_loss', 'c_acc', 'nc_loss', 'nc_acc'))
    for i in range(val_history.__len__()):
        f.write('%.6f  %.6f  %.6f  %.6f  %.6f  %.6f\n' % (
        train_history[i][0], train_history[i][1], val_history[i][0], val_history[i][1], val_history[i][2],
        val_history[i][3]))
    f.close()
    f = open(os.path.join(saver_dir, 'trainLoss.log'), 'a+')
    f.write('focalLoss + 数据均衡\n')
    f.write(
        'modelName=%s, batch_size=%d, learn_rate =%f, epochs=%d, steps_per_epoch=%d, validation_steps=%d, isprep = %d\n' % (
            modelName, batch_size, learn_rate, epochs, steps_per_epoch, validation_steps, isprep))
    f.write('\n%-10s%-10s%-10s\n' % ('iter', 'loss', 'acc'))
    for i in range(train_history_50batch.__len__()):
        f.write('%06d  %.6f  %.6f\n' % ((i + 1) * 50, train_history_50batch[i][0], train_history_50batch[i][1]))
    f.close()
    plot_curv(train_history, val_history, saver_dir)


def training_from_tfrecord(modelName='Mobile', batch_size=100, learn_rate=0.0003, epochs=30, steps_per_epoch=1000,
             validation_steps=500, isprep=False):
    height, width = 64, 64
    with tf.Session() as sess:

        city_image_batch, city_label_batch = next_city_batch(model='train', num_epochs=None,
                                                             batch_size=batch_size//2, out_height=height,
                                                             out_width=width, n_classes=2,
                                                             isProcess=True)
        noncity_image_batch, noncity_label_batch = next_noncity_batch(model='train', num_epochs=None,
                                                                      batch_size=batch_size//2, out_height=height,
                                                                      out_width=width, n_classes=2,
                                                                      isProcess=True)
        val_city_image_batch, val_city_label_batch = next_city_batch(model='val', num_epochs=None,
                                                             batch_size=batch_size, out_height=height,
                                                             out_width=width, n_classes=2,
                                                             isProcess=False)
        val_noncity_image_batch, val_noncity_label_batch = next_noncity_batch(model='val', num_epochs=None,
                                                                      batch_size=batch_size, out_height=height,
                                                                      out_width=width, n_classes=2,
                                                                      isProcess=False)
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # -------------------------------------------------------------------------------------------------------------------
        if modelName == 'Mobile':
            MODEL = MobileNet()
        elif modelName == 'Alexnet':
            MODEL = AlexNet()
        elif modelName == 'Inception':
            MODEL = Inception()
        elif modelName == 'Tinydark':
            MODEL = TinyDark()
        elif modelName == 'Mynet':
            MODEL = MyNet()

        # initiate RMSprop optimizer
        # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
        opt = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Let's train the model using RMSprop
        MODEL.model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
        # ------------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------------------------
        train_history = []
        train_history_50batch = []
        val_history = []
        pre_acc = 0.0
        for i in range(1, epochs + 1):
            # 迭代训练一个epoch
            loss, acc = 0, 0
            for j in range(1, steps_per_epoch + 1):
                c_image_batch, c_label_batch = sess.run([city_image_batch, city_label_batch])
                n_image_batch, n_label_batch = sess.run([noncity_image_batch, noncity_label_batch])
                image_batch = np.concatenate((c_image_batch, n_image_batch), axis=0)
                label_batch = np.concatenate((c_label_batch, n_label_batch), axis=0)
                # show_batch('x',x,False,8,8,1)
                out = MODEL.model.train_on_batch(image_batch, label_batch, class_weight=None, sample_weight=None)
                if j % 50 == 0:
                    train_history_50batch.append(out)
                    print('epoch %d batch iter %d ====> loss = %.6f  acc = %.6f' % (i, j, out[0], out[1]))
                # 测试训练集上性能
                if j > steps_per_epoch - 100:
                    loss += out[0]
                    acc += out[1]
            loss = loss / 100
            acc = acc / 100
            train_history.append([loss, acc])

            # 测试验证集上性能
            val_loss1, val_acc1, val_loss2, val_acc2 = 0, 0, 0, 0
            val_out = []
            for j in range(1, validation_steps + 1):
                c_image_batch, c_label_batch = sess.run([val_city_image_batch, val_city_label_batch])
                n_image_batch, n_label_batch = sess.run([val_noncity_image_batch, val_noncity_label_batch])
                out1 = MODEL.model.evaluate(c_image_batch, c_label_batch, batch_size=batch_size, verbose=0)
                out2 = MODEL.model.evaluate(n_image_batch, n_label_batch, batch_size=batch_size, verbose=0)
                val_loss1 += out1[0]
                val_acc1 += out1[1]
                val_loss2 += out2[0]
                val_acc2 += out2[1]
            val_out.append(val_loss1)
            val_out.append(val_acc1)
            val_out.append(val_loss2)
            val_out.append(val_acc2)
            val_out = np.array(val_out)
            val_out = val_out / validation_steps
            print('epoch %d ====>training acc = %.6f     city acc = %.6f  noncity acc = %.6f' % (
            i, acc, val_out[1], val_out[3]))
            val_history.append(val_out)

        # -------------------------------------------------------------------------------------------------------------------
        times = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '-CE'
        root_dir = 'D:\\MyProject\\Tensorflow\\workspace\\models\\%s' % modelName
        saver_dir = os.path.join(root_dir, times)
        os.mkdir(saver_dir)
        model_name = '%s_model.h5' % modelName
        MODEL.model.save_weights(os.path.join(saver_dir, model_name))

        f = open(os.path.join(saver_dir, 'history.log'), 'a+')
        f.write('focalLoss + 数据均衡\n')
        f.write(
            'modelName=%s, batch_size=%d, learn_rate =%f, epochs=%d, steps_per_epoch=%d, validation_steps=%d, isprep = %d\n' % (
                modelName, batch_size, learn_rate, epochs, steps_per_epoch, validation_steps, isprep))
        f.write('\n%-10s%-10s%-10s%-10s%-10s%-10s\n' % ('t_loss', 't_acc', 'c_loss', 'c_acc', 'nc_loss', 'nc_acc'))
        for i in range(val_history.__len__()):
            f.write('%.6f  %.6f  %.6f  %.6f  %.6f  %.6f\n' % (
            train_history[i][0], train_history[i][1], val_history[i][0], val_history[i][1], val_history[i][2],
            val_history[i][3]))
        f.close()
        f = open(os.path.join(saver_dir, 'trainLoss.log'), 'a+')
        f.write('focalLoss + 数据均衡\n')
        f.write(
            'modelName=%s, batch_size=%d, learn_rate =%f, epochs=%d, steps_per_epoch=%d, validation_steps=%d, isprep = %d\n' % (
                modelName, batch_size, learn_rate, epochs, steps_per_epoch, validation_steps, isprep))
        f.write('\n%-10s%-10s%-10s\n' % ('iter', 'loss', 'acc'))
        for i in range(train_history_50batch.__len__()):
            f.write('%06d  %.6f  %.6f\n' % ((i + 1) * 50, train_history_50batch[i][0], train_history_50batch[i][1]))
        f.close()
        plot_curv(train_history, val_history, saver_dir)
        coord.request_stop()
        coord.join()

if __name__ == "__main__":
    # modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark'  'Mynet']
    # training(modelName='Alexnet', batch_size=64, learn_rate=0.0003, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)
    # training(modelName='Tinydark', batch_size=64, learn_rate=0.0003, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)
    # training(modelName='Inception', batch_size=64, learn_rate=0.0008, epochs=200, steps_per_epoch=1000,
    #          validation_steps=200, isprep=True)
    i=0.8
    training_from_tfrecord(modelName='Mynet', batch_size=128, learn_rate=0.0001*i, epochs=200, steps_per_epoch=500,
             validation_steps=100, isprep=True)
