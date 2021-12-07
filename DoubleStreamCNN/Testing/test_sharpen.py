import os
import threading
import time
import tensorflow as tf

import gdal
import numpy as np
import cv2
from keras import optimizers
from NET.network import PanNet

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

imSize = 10240
smallBlock = 64
fuseImSize = 64
batch_size = imSize//smallBlock

zuobiao = np.zeros([batch_size, 2], int)
fuse_image_batch = np.zeros([batch_size, fuseImSize, fuseImSize, 3], int)
results = []



def classifyDirectly(imagePath ,modelDir):
    row = imSize // smallBlock
    fuseImage = cv2.imread(imagePath)

    global results
    results = np.zeros([imSize//smallBlock, imSize//smallBlock], int)
    # -------------------------------------------------------------------------------------------------------------------
    MODEL = PanNet()
    # initiate RMSprop optimizer
    # opt = optimizers.rmsprop(lr=learn_rate, decay=1e-6)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Let's train the model using RMSprop
    MODEL.model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
    model_name = 'model.h5'
    MODEL.model.load_weights(os.path.join(modelDir, model_name))
    # ------------------------------------------------------------------------------------------------------------------
    for y in range(row):
        fuse_image_batch = np.zeros([batch_size, fuseImSize, fuseImSize, 3], int)
        # 获取批次数据
        for x in range(row):
            fuseTemp = fuseImage[y*fuseImSize:(y+1)*fuseImSize,x*fuseImSize:(x+1)*fuseImSize,:]
            # aaa = fuseTemp[:,:,0]
            # fuseTemp[:, :, 0] = fuseTemp[:,:,2]
            # fuseTemp[:, :, 2] = aaa
            fuse_image_batch[x, :, :, :] = fuseTemp

        fuse_image_batch = fuse_image_batch.astype(np.float32)
        fuse_image_batch = fuse_image_batch / 255.0

        predict = MODEL.model.predict_on_batch([fuse_image_batch])
        print('处理第%d批数据\n' % y)

        for x in range(row):
            if predict[x][0] > predict[x][1]:
                results[y, x] = 1

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
    # modelDir = r'F:\硕士研究生学习\建成区提取\Built-Up Area Detection Contrast Algorithms\Supervised Approaches\DoubleStreamCNN(IEEE_J_Stars)\models\Sharpen'
    #
    # start_time = time.time()
    #
    # imagePath = r"D:\Pictures\sp\109.bmp"
    # print('================================================\n')
    # classifyDirectly(imagePath,modelDir)
    # print("分类结束\n")
    # print("保存图像\n")
    # # 写模式有tif（RGB全图）和bin（二值block结果）两种
    # writeResult('tif',"F:\\DATA\\Remote Sensing\\GF2\\PerformanceTest\\images\\109",'F:\\DATA\\Remote Sensing\\GF2\\PerformanceTest\\SuperviseAlgorithm\\SharpenImage(J-star-revise)\\')
    # writeResult('bin', "F:\\DATA\\Remote Sensing\\GF2\\PerformanceTest\\images\\109", 'F:\\DATA\\Remote Sensing\\GF2\\PerformanceTest\\SuperviseAlgorithm\\SharpenImage(J-star-revise)\\')
    # print("保存图像完成\n")
    # end_time = time.time()
    # print(end_time-start_time)


    # im= cv2.imread('D:/Pictures/DSCNN/origin.bmp')
    # im = im[:,:,1]
    # im_r_f = cv2.imread('D:/Pictures/DSCNN/step1.bmp')
    # im_r_f = im_r_f[:,:,1]
    # gt = cv2.imread('D:/Pictures/DSCNN/gt.bmp')
    # gt = gt[:,:,1]
    #
    # for i in range(5):
    #     for j in range(5):
    #         im_temp = im[i*32:(i+1)*32, j*32:(j+1)*32]
    #         im_r_f_temp = im_r_f[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
    #         gt_temp = gt[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
    #
    #         if np.sum(im_temp != gt_temp) > np.sum(im_r_f_temp != gt_temp):
    #             print("im correct blocks: %d   error blocks: %d " % (
    #             np.sum(im_temp == gt_temp), np.sum(im_temp != gt_temp)))
    #             print(
    #                 "im_r_f correct blocks: %d   error blocks: %d " % (
    #                 np.sum(im_r_f_temp == gt_temp), np.sum(im_r_f_temp != gt_temp)))
    #             print("i: %d    j: %d" % (i, j))
    #         print("==================================================")
    #
    # i,j = 0,2
    # im_temp = im[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
    # im_r_f_temp = im_r_f[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
    # gt_temp = gt[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
    #
    # big1 = cv2.imread('D:/Pictures/DSCNN/part1.bmp')
    # big2 = cv2.imread('D:/Pictures/DSCNN/part1.bmp')
    # big1 = big1[i * 2048:(i + 1) * 2048, j * 2048:(j + 1) * 2048, :]
    # big2 = big2[i * 2048:(i + 1) * 2048, j * 2048:(j + 1) * 2048, :]
    # for i in range(32):
    #     for j in range(32):
    #         if im_temp[i, j] > 125:
    #             big1[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, 0] = 255
    #
    # for i in range(32):
    #     for j in range(32):
    #         if im_r_f_temp[i,j] > 125:
    #             big2[i*64:(i+1)*64, j*64:(j+1)*64,0] = 255
    # cv2.imwrite('D:/Pictures/DSCNN/origin_rgb.bmp', big1)
    # cv2.imwrite('D:/Pictures/DSCNN/step1_rgb.bmp', big2)


    im = cv2.imread('D:/Pictures/DSCNN/origin.bmp')
    big1 = np.zeros([2048,2048,3], np.uint8)
    i, j = 0, 2
    im = im[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32, :]
    for i in range(32):
        for j in range(32):
            if im[i, j,1] > 125:
                big1[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :] = 255

    cv2.imwrite('D:/Pictures/DSCNN/origin_bin_small.bmp', big1)

