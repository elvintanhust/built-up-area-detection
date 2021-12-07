# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from Function.function import show_batch
from Function.function import plot_curv

from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.models import Model

from NET.mynet import MyNet
from NET.FCN import FCN
from LOSS.myloss import FCN_loss
from LOSS.myloss import commission
from LOSS.myloss import omission
from LOSS.myloss import accuracy

from tensorflow.contrib.keras.python.keras.backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

def training(learn_rate=0.0003):
    height, width = 10240, 10240
    FCN_in_H,FCN_in_W = 60,60
    batch_size = 18
    threshold = 0.5
    trainData = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']
    testData = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110']
    with tf.Session() as sess:
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 模型初始化
        MODEL_LMB = MyNet()
        opt1 = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_LMB.model.compile(loss='categorical_crossentropy',
                            optimizer=opt1,
                            metrics=['accuracy'])

        MODEL_FCN = FCN(size=(FCN_in_W, FCN_in_H, 256))
        opt2 = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_FCN.model.compile(loss=FCN_loss,
                                optimizer=opt2,
                                metrics=['accuracy',commission,omission])

        MODEL_FCN_test = FCN(size=(160, 160, 256))
        opt2 = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_FCN_test.model.compile(loss='mean_squared_error',
                                optimizer=opt2,
                                metrics=['accuracy'])
        init_op = tf.local_variables_initializer()
        sess.run(init_op)

        LMB_Model_dir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE\Mynet_model.h5'
        MODEL_LMB.model.load_weights(LMB_Model_dir)
        features_layer = Model(inputs=MODEL_LMB.model.input,outputs=MODEL_LMB.model.layers[55].output)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 提取深度特征
        featureTrain = np.zeros([batch_size, 160, 160, 256], float)
        labelTrain =  np.zeros([batch_size, 160, 160], float)
        for imInd in range(batch_size):
            imName = os.path.join(r'E:\DATA\GF2\PerformanceTest\images', trainData[imInd], 'part1.bmp')
            labelName = os.path.join(r'E:\DATA\GF2\PerformanceTest\BlockAnnotation', trainData[imInd] + '.bmp')
            im = cv2.imread(imName)
            im = im[:, :, 1]
            label = cv2.imread(labelName)
            label = label[:, :, 1]
            label = label / 255
            labelTrain[imInd,:,:] = label
            for i in range(160):
                city_image_batch = np.zeros([160, 64, 64, 1], int)
                for j in range(160):
                    city_image_batch[j, :, :, 0] = im[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
                city_image_batch = city_image_batch.astype(np.float32)
                city_image_batch = city_image_batch / 255
                feature_temp = features_layer.predict_on_batch(city_image_batch)
                featureTrain[imInd,i, :, :] = feature_temp[:, :]
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 模型训练
        for epoch in range(100):
            # -------------------------------------------------------------------------------------------------------------------
            # 训练全卷积神经网络
            for train_iter in range(100):
                input_data = np.zeros([batch_size,FCN_in_W,FCN_in_H,256],float)
                label_batch = np.zeros([batch_size,FCN_in_W,FCN_in_H,1],float)
                for item_in_batch in range(batch_size):
                    posX = np.random.randint(0, 160-FCN_in_W)
                    posY = np.random.randint(0, 160-FCN_in_H)
                    input_data[item_in_batch,:,:,:] = featureTrain[item_in_batch,posX:posX+FCN_in_W,posY:posY+FCN_in_H,:]
                    label_batch[item_in_batch, :, :, 0] = labelTrain[item_in_batch,posX:posX + FCN_in_W, posY:posY + FCN_in_H]
                out = MODEL_FCN.model.train_on_batch(input_data, label_batch, class_weight=None, sample_weight=None)
                if (train_iter+1) % 20 == 0:
                    print('epoch %d  iter %d ====> loss = %.12f  acc = %.6f  虚警：%.6f   漏警：%.6f' % (epoch+1, train_iter+1, out[0], out[1],out[2],out[3]))
            MODEL_FCN.model.save_weights(r'D:\MyProject\Tensorflow\workspace\models\FCN\FCN_model.h5')
            MODEL_FCN_test.model.load_weights(r'D:\MyProject\Tensorflow\workspace\models\FCN\FCN_model.h5')
            # ------------------------------------------------------------------------------------------------------------------
            # 测试
            PT,PF,NT,NF = 0,0,0,0
            for testInd in range(10):
                testImName = os.path.join(r'E:\DATA\GF2\PerformanceTest\images', testData[testInd], 'part1.bmp')
                testLabelName = os.path.join(r'E:\DATA\GF2\PerformanceTest\BlockAnnotation', testData[testInd] + '.bmp')
                im = cv2.imread(testImName)
                im = im[:, :, 1]
                label = cv2.imread(testLabelName)
                label = label[:, :, 1]
                label = label / 255
                features = np.zeros([1,160, 160, 256], float)
                # 提取深度特征
                for i in range(160):
                    city_image_batch = np.zeros([160, 64, 64, 1], int)
                    for j in range(160):
                        city_image_batch[j, :, :, 0] = im[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
                    city_image_batch = city_image_batch.astype(np.float32)
                    city_image_batch = city_image_batch / 255
                    feature_temp = features_layer.predict_on_batch(city_image_batch)
                    features[0,i, :, :] = feature_temp[:, :]
                predicts = MODEL_FCN_test.model.predict(features,batch_size=1,verbose=0)
                predicts.shape = [160,160]
                predicts[predicts <= threshold] = 0.0
                predicts[predicts > threshold]=1.0
                PT = PT + np.sum(np.multiply(predicts==1 ,label==1))
                PF = PF + np.sum(np.multiply(predicts==1 ,label==0))
                NT = NT + np.sum(np.multiply(predicts==0 ,label==0))
                NF = NF + np.sum(np.multiply(predicts==0 ,label==1))
            f = open(r'D:\MyProject\Tensorflow\workspace\models\FCN\test.txt', 'a+')
            overallAcc = (PT+NT)/(PT+NT+PF+NF)*100
            com = PF / (PT + PF + 0.0)*100
            omi = NF / (PT + NF + 0.0)*100
            f.write('准确率：%.2f%%  虚警：%.2f%%  漏警：%.2f%%\n'%(overallAcc,com,omi))
            f.close()
            if overallAcc > 99.5:
                break

        # ------------------------------------------------------------------------------------------------------------------
        coord.request_stop()
        coord.join()

def testing_08(imDir,saveDir):
    height, width = 6400, 6400
    FCN_in_H,FCN_in_W = height//64,width//64
    threshold = 0.5
    voteThreshold = 4

    with tf.Session() as sess:
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 模型初始化
        MODEL_LMB = MyNet()
        opt1 = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_LMB.model.compile(loss='categorical_crossentropy',
                            optimizer=opt1,
                            metrics=['accuracy'])

        MODEL_FCN = FCN(size=(FCN_in_W, FCN_in_H, 256))
        opt2 = optimizers.Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_FCN.model.compile(loss=FCN_loss,
                                optimizer=opt2,
                                metrics=['accuracy',accuracy,commission,omission])


        init_op = tf.local_variables_initializer()
        sess.run(init_op)

        LMB_Model_dir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE\Mynet_model.h5'
        MODEL_LMB.model.load_weights(LMB_Model_dir)
        features_layer = Model(inputs=MODEL_LMB.model.input,outputs=MODEL_LMB.model.layers[55].output)

        MODEL_FCN.model.load_weights(r'D:\MyProject\Tensorflow\workspace\models\FCN\FCN_model.h5')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # ------------------------------------------------------------------------------------------------------------------
        # 测试
        im = cv2.imread(imDir)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im[:, :, 1]
        features = np.zeros([1,FCN_in_H,FCN_in_W, 256], float)
        # 提取深度特征
        for i in range(FCN_in_H):
            city_image_batch = np.zeros([FCN_in_W, 64, 64, 1], int)
            for j in range(FCN_in_W):
                city_image_batch[j, :, :, 0] = im[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
            city_image_batch = city_image_batch.astype(np.float32)
            city_image_batch = city_image_batch / 255
            feature_temp = features_layer.predict_on_batch(city_image_batch)
            features[0,i, :, :] = feature_temp[:, :]
        predicts = MODEL_FCN.model.predict(features,batch_size=1,verbose=0)
        # #------------------------------------------------------------------
        # # 输出8个通道
        # # for i in range(8):
        # #     imtemp = predicts[0,:,:,i]
        # #     max_i = np.max(imtemp)
        # #     min_i = np.min(imtemp)
        # #     imtemp = np.ceil((imtemp-min_i)/(max_i-min_i)*255)
        # #     imtemp = imtemp.astype(np.uint8)
        # #     cv2.imwrite(r'E:\Desktop\FCN_%d.bmp'%(i,),imtemp)
        # #------------------------------------------------------------------
        # # 输出8个通道均值
        # # predicts = np.mean(predicts,axis=3)
        # # predicts.shape = [160,160]
        # # predicts[predicts <= threshold] = 0.0
        # # predicts[predicts > threshold]=1.0
        #
        # #------------------------------------------------------------------
        # # 投票法
        predicts[predicts <= threshold] = 0.0
        predicts[predicts > threshold] = 1.0
        predicts = np.sum(predicts, axis=3)
        predicts.shape = [FCN_in_H, FCN_in_W]
        predicts[predicts <= voteThreshold] = 0.0
        predicts[predicts >voteThreshold] = 1.0

        predicts = predicts*255
        predicts = predicts.astype(np.uint8)
        # cv2.imwrite(saveDir,predicts)
        # 放大图像
        predicts = cv2.resize(predicts,(FCN_in_H*64,FCN_in_W*64))
        predicts = cv2.threshold(predicts,125,255,cv2.THRESH_BINARY)
        predicts = predicts[1]
        predicts = predicts.astype(np.uint8)
        # cv2.imwrite(saveDir, predicts)
        result = np.zeros([height, width,3])
        im[im>253] = 253
        result[:,:,1] = im[:,:]
        result[:, :, 2] = im[:, :]
        predicts[predicts<100] = im[predicts<100]
        # for i in range(height):
        #     for j in range(width):
        #         if(predicts[i, j])>100:
        #             result[i,j, 0] = 255
        #         else:
        #             result[i, j, 0] = im[i,j]
        result[:, :, 0] = predicts[:, :]
        result = result.astype(np.uint8)
        cv2.imwrite(saveDir, result)
        # #------------------------------------------------------------------
        # 单通道
        # predicts[predicts <= threshold] = 0.0
        # predicts[predicts > threshold] = 1.0
        # predicts.shape = [160, 160]
        # predicts = predicts*255
        # predicts = predicts.astype(np.uint8)
        # cv2.imwrite(saveDir,predicts)
        # ------------------------------------------------------------------------------------------------------------------
        coord.request_stop()
        coord.join()

def testing_all(imDir,saveDir):
    height, width = 10240, 10240
    FCN_in_H,FCN_in_W = 160,160
    threshold = 0.5
    voteThreshold = 7

    with tf.Session() as sess:
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 模型初始化
        MODEL_LMB = MyNet()
        opt1 = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_LMB.model.compile(loss='categorical_crossentropy',
                            optimizer=opt1,
                            metrics=['accuracy'])

        MODEL_FCN = FCN(size=(FCN_in_W, FCN_in_H, 256))
        opt2 = optimizers.Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        MODEL_FCN.model.compile(loss=FCN_loss,
                                optimizer=opt2,
                                metrics=['accuracy',accuracy,commission,omission])


        init_op = tf.local_variables_initializer()
        sess.run(init_op)

        LMB_Model_dir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE\Mynet_model.h5'
        MODEL_LMB.model.load_weights(LMB_Model_dir)
        features_layer = Model(inputs=MODEL_LMB.model.input,outputs=MODEL_LMB.model.layers[55].output)

        MODEL_FCN.model.load_weights(r'D:\MyProject\Tensorflow\workspace\models\FCN\FCN_model_best.h5')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # ------------------------------------------------------------------------------------------------------------------
        # 测试
        testData = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110']
        for testInd in range(10):
            testImName = os.path.join(r'E:\DATA\GF2\PerformanceTest\images', testData[testInd], 'part1.bmp')
            print('im:%s'%(testData[testInd],))
            im = cv2.imread(testImName)
            im = im[:, :, 1]
            features = np.zeros([1,160, 160, 256], float)
            # 提取深度特征
            for i in range(160):
                city_image_batch = np.zeros([160, 64, 64, 1], int)
                for j in range(160):
                    city_image_batch[j, :, :, 0] = im[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
                city_image_batch = city_image_batch.astype(np.float32)
                city_image_batch = city_image_batch / 255
                feature_temp = features_layer.predict_on_batch(city_image_batch)
                features[0,i, :, :] = feature_temp[:, :]
            predicts = MODEL_FCN.model.predict(features,batch_size=1,verbose=0)
            # 单通道输出
            # predicts.shape = [160,160]
            # predicts[predicts <= threshold] = 0.0
            # predicts[predicts > threshold]=1.0

            #------------------------------------------------------------------
            # 投票法
            predicts[predicts <= threshold] = 0.0
            predicts[predicts > threshold] = 1.0
            predicts = np.sum(predicts, axis=3)
            predicts.shape = [160, 160]
            predicts[predicts <= voteThreshold] = 0.0
            predicts[predicts >voteThreshold] = 1.0

            predicts = predicts*255
            predicts = predicts.astype(np.uint8)
            cv2.imwrite(os.path.join(r'E:\DATA\GF2\PerformanceTest\LMB-CNN\FCN', testData[testInd] + '.bmp'),predicts)
        # ------------------------------------------------------------------------------------------------------------------
        coord.request_stop()
        coord.join()

if __name__ == "__main__":
    start_time = time.time()
    in_IM = r'E:\DATA\WordView\WorldView3\054168615010_Kalgoorlie_PS_ENH_OR2A\054168615010_01_P001_PSH\Gray_LR.bmp'
    out_IM =  r'E:\DATA\WordView\WorldView3\054168615010_Kalgoorlie_PS_ENH_OR2A\054168615010_01_P001_PSH\result.bmp'
    testing_08(imDir =in_IM ,saveDir =out_IM)
    times = time.time() - start_time
    print(times)