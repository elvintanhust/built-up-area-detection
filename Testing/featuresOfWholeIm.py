import os
import threading
import time
import tensorflow as tf
from NET.mobilenet import MobileNet
from NET.alexnet import AlexNet
from NET.tinydark import TinyDark
from NET.inception import Inception
from NET.mynet import MyNet
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

blockLength = 4096
smallBlock = 64
imSize = 64
batch_size = 64
srcWidth = 0
srcHeight = 0

zuobiao = np.zeros([batch_size, 2], int)
image_batch = np.zeros([batch_size, imSize, imSize, 1], int)
results = []
channels = 32

FEATURESIZE = 2048
features = np.zeros([10240//imSize, 10240//imSize, FEATURESIZE], float)
dataEmpty = False
tifTime = 0

def batchfeature2im(features,imsIn,save_dir):
    b,h,w,c = features.shape
    height,width = 64,64
    col = int(np.sqrt(b))
    row = col
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

def feature2im(features):
    b, h, w, c = features.shape
    height, width = 64, 64
    col = int(np.sqrt(b))
    row = col
    ims = np.zeros([b, height, width, c], np.uint8)
    for i in range(c):
        feature = features[:, :, :, i]
        max_pixel = np.max(feature)
        min_pixel = np.min(feature)
        feature = (feature - min_pixel) / (max_pixel - min_pixel) * 255
        feature = feature.astype(np.uint8)
        for k in range(b):
            im = feature[k, :, :]
            im = cv2.resize(im, (height, width))
            ims[k,:,:,i] = im
    return ims




# 将16位数据线性裁剪拉伸并转为8位
def unit_16_to_8(data):
    height, width = data.shape
    pHistgoram = cv2.calcHist([data], [0], None, [65535], [0.0, 65535.0])

    sumPix = width * height - pHistgoram[0]
    weight_0 = 0.015
    pDataMin = 0
    pDataMax = 0
    while 1:
        n1 = 0
        n2 = 0
        for kk in range(1, 65535):
            n1 = n1 + pHistgoram[kk]
            if n1 > weight_0 * sumPix:
                pDataMin = kk
                break
        for kk in range(65534, 0, -1):
            n2 = n2 + pHistgoram[kk]
            if n2 > weight_0 * sumPix:
                pDataMax = kk
                break
        if (pDataMax - pDataMin > 350) and (weight_0 < 0.04):
            weight_0 = weight_0 + 0.005
        else:
            break
    while 1:
        if pDataMax - pDataMin < 100:
            pDataMax = pDataMax + 50
            pDataMin = pDataMin - 50
        else:
            break
    res = np.zeros([height, width], float)
    data[data > pDataMax] = pDataMax
    data[data < pDataMin] = pDataMin

    res[:, :] = (data[:, :] - pDataMin) / (pDataMax - pDataMin + 0.0) * 255
    res = res.astype(np.uint8)

    # img = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)

    return res

def getImageBatch(dataSingal, processSingal,imagePath):
    global dataEmpty
    global srcHeight
    global srcWidth
    dataset = gdal.Open(imagePath)
    if dataset == None:
        print(imagePath + "掩膜失败，文件无法打开")
        dataEmpty = True
        return
    # 获取图像信息
    srcWidth = dataset.RasterXSize  # 栅格矩阵的列数
    srcHeight = dataset.RasterYSize  # 栅格矩阵的行数

    nX = (srcWidth - 1) // blockLength + 1  # 计算列方向上blockLength大小块数
    nY = (srcHeight - 1) // blockLength + 1  # 计算行方向blockLength大小块数

    grids = blockLength // smallBlock
    # driver = gdal.GetDriverByName("GTiff")
    # dataout = driver.Create(os.path.join(os.path.dirname(imagePath),'result.tif'), srcWidth, srcHeight, 3, gdal.GDT_Byte)

    for y in range(nY):
        for x in range(nX):
            IM_SIZE_x = blockLength
            IM_SIZE_y = blockLength
            if x == (nX - 1):
                IM_SIZE_x = int((srcWidth - 1) % blockLength + 1)
            if y == (nY - 1):
                IM_SIZE_y = int((srcHeight - 1) % blockLength + 1)

            startT = time.time()
            data = dataset.ReadAsArray(xoff=x * blockLength, yoff=y * blockLength, xsize=IM_SIZE_x, ysize=IM_SIZE_y,
                                       buf_xsize=IM_SIZE_x, buf_ysize=IM_SIZE_y, buf_type=gdal.GDT_UInt16)  # 获取数据
            data = unit_16_to_8(data)
            # for i in range(3):
            #     dataout.GetRasterBand(i + 1).WriteArray(data[:,:,i],xoff=x * blockLength, yoff=y * blockLength)
            global tifTime
            tifTime = tifTime+time.time()-startT

            # 获取批次数据
            xNum = (IM_SIZE_x - 1) // smallBlock + 1
            yNum = (IM_SIZE_y - 1) // smallBlock + 1
            ka = 0
            kb = 0

            while ka < xNum * yNum:
                global zuobiao
                global image_batch
                zuobiao = np.zeros([batch_size, 2], int)  # 记录坐标数组置-1
                zuobiao[:, :] = -1
                image_batch = np.zeros([batch_size, imSize, imSize, 1], int)

                predict_batch_now = 0
                while predict_batch_now < batch_size:
                    if (ka + kb) == xNum * yNum:
                        break;
                    ia = (ka + kb) // xNum
                    ja = (ka + kb) % xNum
                    imagePartX = smallBlock
                    imagePartY = smallBlock
                    # 行列末尾小块处理
                    if ia == yNum - 1:
                        imagePartY = (IM_SIZE_y - 1) % smallBlock + 1
                        if imagePartY < 30:
                            kb = kb + 1
                            continue
                    if ja == xNum - 1:
                        imagePartX = (IM_SIZE_x - 1) % smallBlock + 1
                        if imagePartX < 30:
                            kb = kb + 1
                            continue
                    imagePart = data[(ia * smallBlock):(ia * smallBlock + imagePartY),
                                (ja * smallBlock):(ja * smallBlock + imagePartX)]
                    flag_pixel_hist = True
                    if x == 0 or y == 0 or x == (nX - 1) or y == (nY - 1):
                        hist_temp = cv2.calcHist([imagePart], [0], None, [256], [0.0, 255.0])
                        if np.sum(hist_temp > 5) < 10:
                            flag_pixel_hist = False
                    ave = np.average(imagePart[:, :])
                    if 1 or(ave >= 20 and ave <= 230 and flag_pixel_hist):
                        imagePart = cv2.resize(imagePart, (imSize, imSize))
                        image_batch[predict_batch_now, :, :, 0] = imagePart
                        zuobiao[predict_batch_now, 0] = ia + y * grids  # 纵坐标
                        zuobiao[predict_batch_now, 1] = ja + x * grids  # 横坐标
                        predict_batch_now = predict_batch_now + 1
                    kb = kb + 1

                dataSingal.set()  # 数据准备好了
                processSingal.wait()  # 等待处理线程读取数据
                dataSingal.clear()  # 数据读完后重新准备数据

                ka = ka + kb
                kb = 0
    del dataset
    # del dataout
    dataEmpty = True
    return


def getFeatures(dataSingal, processSingal,modelName,modelDir):
    height, width = imSize, imSize
    num_classes = 2
    global srcHeight
    global srcWidth
    dataset = gdal.Open(imagePath)
    if dataset == None:
        print(imagePath + "掩膜失败，文件无法打开")
        return
    # 获取图像信息
    srcWidth = dataset.RasterXSize  # 栅格矩阵的列数
    srcHeight = dataset.RasterYSize  # 栅格矩阵的行数
    del dataset

    nX_s = (srcWidth - 1) // smallBlock + 1  # 计算列方向上smallBlock大小块数
    nY_s = (srcHeight - 1) // smallBlock + 1  # 计算行方向smallBlock大小块数

    global results
    global features
    results = np.zeros([channels, srcHeight, srcWidth], np.uint8)
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
    # 取某一层的输出为输出新建为model，采用函数模型
    predict_layer = Model(inputs=MODEL.model.input,
                               outputs=MODEL.model.layers[6].output)

    # ------------------------------------------------------------------------------------------------------------------
    k = 1
    while not dataEmpty:
        dataSingal.wait()  # 等待数据准备
        imagesToClassify = image_batch
        zuobiaoTemp = zuobiao
        processSingal.set()  # 数据取完，可以继续准备数据
        cv2.waitKey(1)
        processSingal.clear()  # 下一批数据准备好后阻塞数据线程

        imagesToClassify = imagesToClassify.astype(np.float32)
        imagesToClassify = imagesToClassify / 255
        # ims = show_batch('show', imagesToClassify, showlabel=False, col=8, row=8, channel=1, predicts=None, labels=None)
        feature = predict_layer.predict_on_batch( imagesToClassify)
        feature = feature2im(feature)
        # predict = MODEL.model.predict_on_batch(imagesToClassify)
        print('处理第%d批数据\n' % k)
        k = k + 1
        batch_size_now = batch_size - np.sum(zuobiaoTemp[:, 0] == -1)

        for bb in range(batch_size_now):
            y = zuobiaoTemp[bb, 0]
            x = zuobiaoTemp[bb, 1]
            for kk in range(channels):
                results[kk,y*smallBlock:(y+1)*smallBlock,x*smallBlock:(x+1)*smallBlock] = feature[bb,:,:,kk]


if __name__ == "__main__":
    modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark'  'Mynet']
    modelName = 'Mynet'
    modelDir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'

    imInd = '01'
    imagePath = os.path.join(r'E:\DATA\GF2\PerformanceTest\images', imInd)
    imagePath = os.path.join(imagePath, 'part1.tif')

    featuresDir = os.path.join(r'E:\DATA\GF2\PerformanceTest\MyNet', imInd)
    featuresDir = os.path.join(featuresDir,'CNNClassify')
    print("start\n")
    start_time = time.time()

    dataSingal = threading.Event()
    processSingal = threading.Event()

    tIM = threading.Thread(target=getImageBatch, args=(dataSingal, processSingal,imagePath))
    tCL = threading.Thread(target=getFeatures, args=(dataSingal, processSingal,modelName,modelDir))

    tIM.start()
    tCL.start()
    tIM.join()
    tCL.join()
    times = time.time() - start_time
    minu, sec = divmod(times, 60)
    hour, minu = divmod(minu, 60)
    print('分类总计耗时：%02d:%02d:%02d\n' % (hour, minu, sec))
    global results
    i = 4
    imname = os.path.join(featuresDir,'channel_' + str(i) + '.bmp')
    im = results[i,:,:]
    cv2.imwrite(imname,im)


    featuresDir = os.path.join(r'E:\DATA\GF2\PerformanceTest\MyNet', imInd)
    featuresDir = os.path.join(featuresDir, 'CNNClassify')
    imIOR = cv2.imread(r'E:\DATA\GF2\PerformanceTest\MyNet\106\CNNClassify\106.tif')
    imIOR = imIOR[:, :, 0]
    i = 4
    imname = os.path.join(featuresDir, 'channel_' + str(i) + '.bmp')
    im = cv2.imread(imname)
    im = im[:,:,0]
    height, width = im.shape
    w = 0
    h = 0
    while w < width:
        if (w - 1) > 0:
            im[:, w - 1] = 0
        im[:, w] = 0
        if (w + 1) < width:
            im[:, w + 1] = 0
        w = w + 64
    while h < height:
        if (h - 1) > 0:
            im[h - 1, :] = 0
        im[h, :] = 0
        if (h + 1) < height:
            im[h + 1, :] = 0
        h = h + 64
    im[imIOR<255] = 0
    t = 100
    im[im > t] = 255
    im[im <= t] = 0
    cv2.imwrite(imname,im)
