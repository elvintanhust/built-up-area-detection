import os
import threading
import time
import tensorflow as tf


import gdal
import numpy as np
import cv2
from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.models import Model
from Function.function import show_batch
from sklearn.cluster import KMeans
from Function.function import getDataByList
from Function.TFRecord2image import next_city_batch
from Function.TFRecord2image import next_noncity_batch

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim


def batchfeature2im(features,imsIn,save_dir,height,width,col,row):
    b,h,w,c = features.shape


    ims = np.zeros([height * col, width * row], np.uint8)
    root_dir = save_dir
    cv2.imwrite( os.path.join(root_dir,'in.bmp'), imsIn)
    for i in range(c):
        feature = features[:,:,:,i]
        max_pixel = np.max(feature)
        min_pixel = np.min(feature)
        feature = (feature-min_pixel)/(max_pixel-min_pixel)*255
        feature = feature.astype(np.uint8)
        for x in range(row):
            for y in range(col):
                im = feature[x*col+y,:,:]
                im = cv2.resize(im,(height,width))
                ims[(x * height):(x * height + height), (y * width):(y * width + width)] = im
        names = str(i) + '.bmp'
        save_dir = os.path.join(root_dir,names)
        cv2.imwrite(save_dir,ims)


def visualization_batch(modelName,modelDir,save_dir):
    height, width = 64,64
    with tf.Session() as sess:
        batch_size = 64

        city_image_batch, city_label_batch = next_city_batch(model='train', num_epochs=None,
                                                             batch_size=batch_size // 2, out_height=height,
                                                             out_width=width, n_classes=2,
                                                             isProcess=True)
        noncity_image_batch, noncity_label_batch = next_noncity_batch(model='train', num_epochs=None,
                                                                      batch_size=batch_size // 2, out_height=height,
                                                                      out_width=width, n_classes=2,
                                                                      isProcess=True)
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

        opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        MODEL.model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
        model_name = '%s_model.h5' % modelName
        MODEL.model.load_weights(os.path.join(modelDir, model_name))
        # 已有的model在load权重过后
        # ------------------------------------------------------------------------------------------------------------------
        # 取某一层的输出为输出新建为model，采用函数模型
        predict_layer = Model(inputs=MODEL.model.input,
                              outputs=MODEL.model.layers[4].output) # 32 * 32 * 64

        c_image_batch, c_label_batch = sess.run([city_image_batch, city_label_batch])
        n_image_batch, n_label_batch = sess.run([noncity_image_batch, noncity_label_batch])
        image_batch = np.concatenate((c_image_batch, n_image_batch), axis=0)
        ims = show_batch('show', image_batch, showlabel=False, col=8, row=8, channel=1, predicts=None, labels=None)
        feature = predict_layer.predict_on_batch(image_batch)
        batchfeature2im(feature, ims, os.path.join(save_dir,'04'),32,32,8, 8)
        # ------------------------------------------------------------------------------------------------------------------
        # 取某一层的输出为输出新建为model，采用函数模型
        predict_layer = Model(inputs=MODEL.model.input,
                              outputs=MODEL.model.layers[24].output) # 16 * 16 * 64

        feature = predict_layer.predict_on_batch(image_batch)
        batchfeature2im(feature, ims, os.path.join(save_dir,'24'),32,32,8, 8)
        # ------------------------------------------------------------------------------------------------------------------
        # 取某一层的输出为输出新建为model，采用函数模型
        predict_layer = Model(inputs=MODEL.model.input,
                              outputs=MODEL.model.layers[38].output) # 8 * 8 * 96

        feature = predict_layer.predict_on_batch(image_batch)
        batchfeature2im(feature, ims, os.path.join(save_dir,'38'),32,32,8, 8)
        # ------------------------------------------------------------------------------------------------------------------
        # 取某一层的输出为输出新建为model，采用函数模型
        predict_layer = Model(inputs=MODEL.model.input,
                              outputs=MODEL.model.layers[52].output) # 4 * 4 * 128

        feature = predict_layer.predict_on_batch(image_batch)
        batchfeature2im(feature, ims, os.path.join(save_dir,'52'),32,32,8, 8)
        coord.request_stop()
        coord.join()

def visualization_params(modelName,modelDir,save_dir):
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

    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
    model_name = '%s_model.h5' % modelName
    MODEL.model.load_weights(os.path.join(modelDir, model_name))
    # 已有的model在load权重过后
    rc = [[4,8],[32,64],[8,8],[32,64],[8,8],[64,96],[8,12],[96,128]]
    lay = [1, 4, 8,13,26,29,40,43]
    for i in range(8):  # save the BN params
        param = MODEL.model.layers[lay[i]].get_weights()
        kernel_param = param[0] #get the kernel
        print(kernel_param.shape)
        max_p = np.max(kernel_param)
        min_p = np.min(kernel_param)
        kernel_param = (kernel_param-min_p)/(max_p-min_p+0.0)*255
        kernel_param=kernel_param.astype(np.uint8)
        _,_,c1,c2 = kernel_param.shape
        r,c = rc[i]
        res = np.zeros([r * 5 + r - 1, c * 5 + c - 1], np.uint8)
        if c1 == 1:
            kk = 0
            for ir in range(r):
                for ic in range(c):
                    res[ir*5+ir:(ir+1)*5+ir,ic*5+ic:(ic+1)*5 + ic] = cv2.resize(kernel_param[:,:,:,kk].reshape([3,3]),(5,5))
                    kk += 1
        if c2 == 1:
            kk = 0
            for ir in range(r):
                for ic in range(c):
                    res[ir*5+ir:(ir+1)*5+ir,ic*5+ic:(ic+1)*5 + ic] = cv2.resize(kernel_param[:,:,kk,:].reshape([3,3]),(5,5))
                    kk += 1
        if c2 != 1 and c1 != 1:
            kk = 0
            for ir in range(r):
                for ic in range(c):
                    res[ir*5+ir:(ir+1)*5+ir,ic*5+ic:(ic+1)*5 + ic] = cv2.resize(kernel_param[:,:,ir,ic].reshape([3,3]),(5,5))
                    kk += 1
        name = str(i) + '.jpg'
        cv2.imwrite(os.path.join(save_dir,name),res)

def features_batch(modelName,modelDir,save_dir):
    height, width = 64,64
    with tf.Session() as sess:
        batch_size = 100

        city_image_batch, city_label_batch = next_city_batch(model='test', num_epochs=1,
                                                             batch_size=batch_size, out_height=height,
                                                             out_width=width, n_classes=2,
                                                             isProcess=False)
        noncity_image_batch, noncity_label_batch = next_noncity_batch(model='test', num_epochs=1,
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

        opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        MODEL.model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
        model_name = '%s_model.h5' % modelName
        MODEL.model.load_weights(os.path.join(modelDir, model_name))
        # 已有的model在load权重过后
        # ------------------------------------------------------------------------------------------------------------------
        # 取某一层的输出为输出新建为model，采用函数模型
        predict_layer = Model(inputs=MODEL.model.input,
                              outputs=MODEL.model.layers[55].output)

        featuresFile = open(os.path.join(save_dir,'city.txt'),'a+')
        while 1:
            try:
                image_batch, label_batch = sess.run([city_image_batch, city_label_batch])
            except tf.errors.OutOfRangeError:
                break
            feature = predict_layer.predict_on_batch(image_batch)
            for i in range(batch_size):
                for j in range(256):
                    featuresFile.write('%f '%(feature[i,j],))
                featuresFile.write('\n')
        featuresFile.close()

        featuresFile = open(os.path.join(save_dir, 'noncity.txt'), 'a+')
        while 1:
            try:
                image_batch, label_batch = sess.run([noncity_image_batch, noncity_label_batch])
            except tf.errors.OutOfRangeError:
                break
            feature = predict_layer.predict_on_batch(image_batch)
            for i in range(batch_size):
                for j in range(256):
                    featuresFile.write('%f '%(feature[i,j],))
                featuresFile.write('\n')
        featuresFile.close()
        # ------------------------------------------------------------------------------------------------------------------

        coord.request_stop()
        coord.join()


if __name__ == "__main__":
    # modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark'  'Mynet']
    if 0:
        modelName = 'Mynet'
        modelDir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'
        imagePath = r'E:\DATA\GF2\GF2测试\01GF2_PMS1_E119.2_N31.4_20150211_L1A0000645651江苏\part1.tif'
        featuresDir = r'D:\MyProject\GraduationProject\DATA\features\js'
        print("start\n")

    if 0:
        modelName = 'Mynet'
        modelDir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'
        featuresDir = r'D:\MyProject\GraduationProject\DATA\features\visualizationBatch'
        print("start\n")
        # visualization_params(modelName, modelDir, featuresDir)
        visualization_batch(modelName, modelDir, featuresDir)
        print("end\n")

    if 1:
        modelName = 'Mynet'
        modelDir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'
        featuresDir = r'E:\DATA\GF2\PerformanceTest\CNNfeatures'
        print("start\n")
        features_batch(modelName, modelDir, featuresDir)
        print("end\n")


