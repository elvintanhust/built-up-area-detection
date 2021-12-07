import os
import threading
import time
import tensorflow as tf

import gdal
import numpy as np
import cv2
from keras import optimizers
from NET.network import DoubleStreamNet

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

imSize = 10240
smallBlock = 64
panImSize = 64
mulImSize = 16
batch_size = imSize//smallBlock

zuobiao = np.zeros([batch_size, 2], int)
pan_image_batch = np.zeros([batch_size, panImSize, panImSize, 1], int)
mul_image_batch = np.zeros([batch_size, mulImSize, mulImSize, 3], int)
results = []

dataEmpty = False
tifTime = 0

def getImageBatch(dataSingal, processSingal,imagePath):
    global dataEmpty
    row = imSize//smallBlock
    panImage = cv2.imread(os.path.join(imagePath,'part1.bmp'))
    panImage = panImage[:,:,0]
    mulImage = cv2.imread(os.path.join(imagePath,'ortho_m.bmp'))
    for y in range(row):
        print(y)
        global zuobiao
        global pan_image_batch
        global mul_image_batch
        zuobiao = np.zeros([batch_size, 2], int)  # 记录坐标数组置-1
        zuobiao[:, :] = -1
        pan_image_batch = np.zeros([batch_size, panImSize, panImSize, 1], int)
        mul_image_batch = np.zeros([batch_size, mulImSize, mulImSize, 3], int)
        # 获取批次数据
        for x in range(row):
            panTemp = panImage[y*panImSize:(y+1)*panImSize,x*panImSize:(x+1)*panImSize]
            mulTemp = mulImage[y*mulImSize:(y+1)*mulImSize,x*mulImSize:(x+1)*mulImSize,:]
            zuobiao[x, 0] = y  # 纵坐标
            zuobiao[x, 1] = x  # 横坐标
            pan_image_batch[x, :, :, 0] = panTemp
            mul_image_batch[x, :, :, :] = mulTemp
        dataSingal.set()  # 数据准备好了
        processSingal.wait()  # 等待处理线程读取数据
        dataSingal.clear()  # 数据读完后重新准备数据
    dataEmpty = True
    return

def classify(dataSingal, processSingal,modelDir):
    panHeight, panWidth = panImSize, panImSize
    mulHeight, mulWidth = mulImSize, mulImSize
    num_classes = 2

    global results
    results = np.zeros([imSize//smallBlock, imSize//smallBlock], int)
    # -------------------------------------------------------------------------------------------------------------------
    MODEL = DoubleStreamNet()
    # initiate RMSprop optimizer
    # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
    model_name = 'DoubleStreamNet_model.h5'
    MODEL.model.load_weights(os.path.join(modelDir, model_name))
    # ------------------------------------------------------------------------------------------------------------------
    k = 1
    while not dataEmpty:
        dataSingal.wait()  # 等待数据准备
        panImagesBatch = pan_image_batch
        mulImagesBatch = mul_image_batch
        zuobiaoTemp = zuobiao
        processSingal.set()  # 数据取完，可以继续准备数据
        cv2.waitKey(1)
        processSingal.clear()  # 下一批数据准备好后阻塞数据线程

        # for i in range(batch_size):
        #     im = imagesToClassify[i,:,:,:]
        #     im = im.astype(np.uint8)
        #     cv2.imshow('show',im)
        #     cv2.waitKey(1000)

        panImagesBatch = panImagesBatch.astype(np.float32)
        panImagesBatch = panImagesBatch / 255
        mulImagesBatch = mulImagesBatch.astype(np.float32)
        mulImagesBatch = mulImagesBatch / 255
        predict = MODEL.model.predict_on_batch([panImagesBatch,mulImagesBatch])
        print('处理第%d批数据\n' % k)
        k = k + 1
        batch_size_now = batch_size - np.sum(zuobiaoTemp[:, 0] == -1)

        for bb in range(imSize//smallBlock):
            if predict[bb][0] > predict[bb][1]:
                results[zuobiaoTemp[bb, 0], zuobiaoTemp[bb, 1]] = 1


def writeResult(model,imagePath, save_dir):
    nY_s, nX_s = results.shape
    if model == 'tif':
        panImage = cv2.imread(os.path.join(imagePath,'part1.bmp'))
        panImage = panImage[:,:,0]
        panImage[panImage >= 253] = 253
        outImage = np.zeros([imSize, imSize, 3], np.uint8)
        for i in range(3):
           outImage[:, :, i] = panImage[:,:]
        for y in range(imSize//smallBlock):
            for x in range(imSize//smallBlock):
                res = results[y,x]
                if res == 1:
                    outImage[y*smallBlock:(y+1)*smallBlock,x*smallBlock:(x+1)*smallBlock,0] = 255
        cv2.imwrite(save_dir+'result.bmp',outImage)
        # for i in range(5):
        #     ry = np.random.randint(0,10240-2048)
        #     rx = np.random.randint(0,10240-2048)
        #     imPart = outImage[ry:ry+2048,rx:rx+2048,:]
        #     cv2.imwrite(os.path.join(save_dir, str(i)+'.bmp'),imPart)
    elif model == 'bin':
        im = results*255
        im = im.astype(np.uint8)
        im_dir = save_dir+'bin.bmp'
        res = np.zeros([nY_s, nX_s,3],np.uint8)
        for i in range(3):
            res[:,:,i] = im
        flag = cv2.imwrite(im_dir,res)
        if flag == False:
            print('写图像失败\n')


if __name__ == "__main__":
    modelDir = r'F:\硕士研究生学习\建成区提取\Built-Up Area Detection Contrast Algorithms\Supervised Approaches\DoubleStreamCNN(IEEE_J_Stars)\models\DoubleStreamNet\2018-01-25_17-17-36-CE'

    start_time = time.time()

    imagePath = r"F:\DATA\Remote Sensing\GF2\PerformanceTest\images\18"
    print('================================================\n')
    dataSingal = threading.Event()
    processSingal = threading.Event()

    tIM = threading.Thread(target=getImageBatch, args=(dataSingal, processSingal,imagePath))
    tCL = threading.Thread(target=classify, args=(dataSingal, processSingal,modelDir))

    tIM.start()
    tCL.start()
    tIM.join()
    tCL.join()

    print("分类结束\n")
    print("保存图像\n")
    # 写模式有tif（RGB全图）和bin（二值block结果）两种
    writeResult('tif',imagePath,'D:\\Pictures\\DSCNN\\')
    writeResult('bin', imagePath, 'D:\\Pictures\\DSCNN\\')
    print("保存图像完成\n")
    end_time = time.time()
    print(end_time-start_time)



    # modelDir = r'D:\MyProject\IEEE_J_Stars1\models\DoubleStreamNet\2018-01-25_17-17-36-CE'
    # ims = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
    # start_time = time.time()
    # imInd = '101'
    #
    # imagePath = os.path.join(r'E:\DATA\GF2\PerformanceTest\images',imInd)
    # print('================================================\n')
    # print("分类开始 序号：%s\n" % (imInd,) )
    # dataSingal = threading.Event()
    # processSingal = threading.Event()
    #
    # tIM = threading.Thread(target=getImageBatch, args=(dataSingal, processSingal,imagePath))
    # tCL = threading.Thread(target=classify, args=(dataSingal, processSingal,modelDir))
    #
    # tIM.start()
    # tCL.start()
    # tIM.join()
    # tCL.join()
    #
    # print("分类结束\n")
    # print("保存图像\n")
    # # 写模式有tif（RGB全图）和bin（二值block结果）两种
    # # writeResult('tif',imagePath,os.path.join(r'E:\DATA\GF2\PerformanceTest\SuperviseAlgorithm\DoubleStreamNet',imInd))
    # # writeResult('bin', imagePath, 'E:\\DATA\\GF2\\PerformanceTest\\SuperviseAlgorithm\\DoubleStreamNet\\'+imInd)
    # print("保存图像完成\n")
    # end_time = time.time()
    # print(end_time-start_time)




