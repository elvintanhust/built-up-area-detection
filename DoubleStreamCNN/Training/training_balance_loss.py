# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from Function.function import show_batch
from Function.function import plot_curv
from Function.function import getDataByList
from Function.TFRecord2image import next_city_batch
from Function.TFRecord2image import next_noncity_batch

from keras import optimizers

from NET.network import PanNet, MulNet, DoubleStreamNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def printParams(model,layer):
    param = model.layers[layer].get_weights()
    kernel_param = param[0]  # get the kernel
    print(kernel_param[:,:,0,0])

def training(modelName='PanNet', batch_size=100, learn_rate=0.0003, epochs=30, steps_per_epoch=1000,
             validation_steps=500, isprep=False):
    panheight = 64
    panwidth = 64
    mulheight = 16
    mulwidth = 16

    with tf.Session() as sess:
        pan_city_image_batch, mul_city_image_batch, city_label_batch = next_city_batch(model='train', num_epochs=None,
                                                                                       batch_size=batch_size // 2,
                                                                                       pan_height=panheight,
                                                                                       pan_width=panwidth,
                                                                                       mul_height=mulheight,
                                                                                       mul_width=mulwidth, n_classes=2,
                                                                                       isProcess=isprep)
        pan_noncity_image_batch, mul_noncity_image_batch, noncity_label_batch = next_noncity_batch(model='train',
                                                                                                   num_epochs=None,
                                                                                                   batch_size=batch_size // 2,
                                                                                                   pan_height=panheight,
                                                                                                   pan_width=panwidth,
                                                                                                   mul_height=mulheight,
                                                                                                   mul_width=mulwidth,
                                                                                                   n_classes=2,
                                                                                                   isProcess=isprep)
        pan_val_city_image_batch, mul_val_city_image_batch, val_city_label_batch = next_city_batch(model='val',
                                                                                                   num_epochs=None,
                                                                                                   batch_size=batch_size,
                                                                                                   pan_height=panheight,
                                                                                                   pan_width=panwidth,
                                                                                                   mul_height=mulheight,
                                                                                                   mul_width=mulwidth,
                                                                                                   n_classes=2,
                                                                                                   isProcess=False)
        pan_val_noncity_image_batch, mul_val_noncity_image_batch, val_noncity_label_batch = next_noncity_batch(
            model='val', num_epochs=None,
            batch_size=batch_size, pan_height=panheight,
            pan_width=panwidth, mul_height=mulheight,
            mul_width=mulwidth, n_classes=2,
            isProcess=False)
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # -------------------------------------------------------------------------------------------------------------------

        if modelName == 'PanNet':
            MODEL = PanNet()
        elif modelName == 'MulNet':
            MODEL = MulNet()
        elif modelName == 'DoubleStreamNet':
            MODEL = DoubleStreamNet()

        if True:
            pan_model = r'D:\MyProject\IEEE_J_Stars1\models\PanNet\2018-01-24_17-39-55-CE\PanNet_model.h5'
            mul_model = r'D:\MyProject\IEEE_J_Stars1\models\MulNet\2018-01-24_17-13-23-CE\MulNet_model.h5'
            MODEL.model.load_weights(pan_model, by_name=True)
            MODEL.model.load_weights(mul_model, by_name=True)
        # initiate RMSprop optimizer
        # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
        for i in range(129):
            MODEL.model.layers[i].trainable = False

        opt = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Let's train the model using RMSprop
        MODEL.model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])


        # ------------------------------------------------------------------------------------------------------------------
        times = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '-CE'
        root_dir = 'D:\\MyProject\\IEEE_J_Stars1\\models\\%s' % modelName
        saver_dir = os.path.join(root_dir, times)
        os.mkdir(saver_dir)
        model_name = '%s_model.h5' % modelName
        train_history = []
        train_history_50batch = []
        val_history = []

        for i in range(1, epochs + 1):
            # 迭代训练一个epoch
            loss, acc = 0, 0
            for j in range(1, steps_per_epoch + 1):
                c_pan_image_batch, c_mul_image_batch, c_label_batch = sess.run(
                    [pan_city_image_batch, mul_city_image_batch, city_label_batch])
                n_pan_image_batch, n_mul_image_batch, n_label_batch = sess.run(
                    [pan_noncity_image_batch, mul_noncity_image_batch, noncity_label_batch])
                label_batch = np.concatenate((c_label_batch, n_label_batch), axis=0)
                # show_batch('x',mul_batch,False,8,16,64,64,3)
                if modelName == 'PanNet':
                    pan_image_batch = np.concatenate((c_pan_image_batch, n_pan_image_batch), axis=0)
                    out = MODEL.model.train_on_batch(pan_image_batch, label_batch, class_weight=None,
                                                     sample_weight=None)
                    # printParams(MODEL.model, 5)
                elif modelName == 'MulNet':
                    mul_image_batch = np.concatenate((c_mul_image_batch, n_mul_image_batch), axis=0)
                    out = MODEL.model.train_on_batch(mul_image_batch, label_batch, class_weight=None,
                                                     sample_weight=None)
                else:
                    pan_image_batch = np.concatenate((c_pan_image_batch, n_pan_image_batch), axis=0)
                    mul_image_batch = np.concatenate((c_mul_image_batch, n_mul_image_batch), axis=0)
                    out = MODEL.model.train_on_batch([pan_image_batch, mul_image_batch], label_batch, class_weight=None,
                                                     sample_weight=None)
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
                c_pan_image_batch, c_mul_image_batch, c_label_batch = sess.run(
                    [pan_val_city_image_batch, mul_val_city_image_batch, val_city_label_batch])
                n_pan_image_batch, n_mul_image_batch, n_label_batch = sess.run(
                    [pan_val_noncity_image_batch, mul_val_noncity_image_batch, val_noncity_label_batch])
                if modelName == 'PanNet':
                    out1 = MODEL.model.evaluate(c_pan_image_batch, c_label_batch, batch_size=batch_size, verbose=0)
                    out2 = MODEL.model.evaluate(n_pan_image_batch, n_label_batch, batch_size=batch_size, verbose=0)
                elif modelName == 'MulNet':
                    out1 = MODEL.model.evaluate(c_mul_image_batch, c_label_batch, batch_size=batch_size, verbose=0)
                    out2 = MODEL.model.evaluate(n_mul_image_batch, n_label_batch, batch_size=batch_size, verbose=0)
                else:
                    out1 = MODEL.model.evaluate([c_pan_image_batch, c_mul_image_batch], c_label_batch,
                                                batch_size=batch_size, verbose=0)
                    out2 = MODEL.model.evaluate([n_pan_image_batch, n_mul_image_batch], n_label_batch,
                                                batch_size=batch_size, verbose=0)
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
            if len(val_history) > 0:
                pre_val_out = val_history[-1]
            else:
                pre_val_out = [0,0,0,0]

            if val_out[1] >0.99 and val_out[3] > 0.99 :
                out_name = str(i) + model_name
                MODEL.model.save_weights(os.path.join(saver_dir, out_name))

            if val_out[1] < 0.05 or val_out[3] < 0.05:
                return False
            val_history.append(val_out)

        # -------------------------------------------------------------------------------------------------------------------
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
        return True


if __name__ == "__main__":
    # modelName = ['PanNet'  'MulNet'  'DoubleStreamNet']
    # training(modelName='MulNet', batch_size=128, learn_rate=0.0001, epochs=40, steps_per_epoch=500,
    #          validation_steps=100, isprep=False)
    # training(modelName='PanNet', batch_size=128, learn_rate=0.0003, epochs=40, steps_per_epoch=500,
    #          validation_steps=100, isprep=False)
    while 1:
        if training(modelName='DoubleStreamNet', batch_size=128, learn_rate=0.00008, epochs=20, steps_per_epoch=200,
             validation_steps=100, isprep=False):
            break
