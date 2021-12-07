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
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

blockLength = 4096
smallBlock = 64
imSize = 64
batch_size = 64
srcWidth = 0
srcHeight = 0
FEATURESIZE = 256

zuobiao = np.zeros([batch_size, 2], int)
image_batch = np.zeros([batch_size, imSize, imSize, 1], int)
results = []
iscity = []
channels = 32

dataEmpty = False
tifTime = 0

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
    nX = (srcWidth - 1) // blockLength + 1  # 计算列方向上blockLength大小块数
    nY = (srcHeight - 1) // blockLength + 1  # 计算行方向blockLength大小块数
    nX_s = (srcWidth - 1) // smallBlock + 1  # 计算列方向上smallBlock大小块数
    nY_s = (srcHeight - 1) // smallBlock + 1  # 计算行方向smallBlock大小块数

    global results
    global iscity
    results = np.zeros([nY_s, nX_s, FEATURESIZE], np.float)
    iscity = np.zeros([nY_s, nX_s], bool)
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
                               outputs=MODEL.model.get_layer('feature').output)
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
        predict = MODEL.model.predict_on_batch(imagesToClassify)
        print('处理第%d批数据\n' % k)
        k = k + 1
        batch_size_now = batch_size - np.sum(zuobiaoTemp[:, 0] == -1)

        for bb in range(batch_size_now):
            y = zuobiaoTemp[bb, 0]
            x = zuobiaoTemp[bb, 1]
            results[y,x,:] = feature[bb,:]
            if predict[bb][0]>predict[bb][1]:
                iscity[y,x] = True

def PCA_im(features):
    rc = int(10240/64)
    X = np.zeros([rc*rc,FEATURESIZE],np.float)
    for i in range(rc):
        for j in range(rc):
            ind = i*rc + j
            X[ind,:] = results[i,j,:]
    pca = PCA(n_components=3, copy=False, whiten=False)
    newX = pca.fit_transform(X)
    im = np.zeros([rc,rc,3],float)
    for i in range(rc):
        for j in range(rc):
            ind = i*rc + j
            im[i, j, :] = newX[ind,:]
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im-im_min)/(im_max - im_min +0.0)*255
    im = im.astype(np.uint8)
    return im

def block_similarity(data):
    global results
    global iscity
    rc = int(10240/64)
    varianceSquared,count = 0.0, 0
    im = np.zeros([10240,10240,3],np.uint8)
    im[:,:,0] = data
    im[:,:,1] = data
    im[:,:,2] = data
    dis = []
    for i in range(rc):
        for j in range(rc):
            featureij = results[i,j,:]
            if i != (rc-1):
                featureneig = results[i+1,j,:]
                varianceSquared += np.sum(np.square(featureij - featureneig))
                dis.append(np.sum(np.square(featureij - featureneig)))
                count += 1
            if j != (rc-1):
                featureneig = results[i,j+1,:]
                varianceSquared += np.sum(np.square(featureij - featureneig))
                dis.append(np.sum(np.square(featureij - featureneig)))
                count += 1
    varianceSquared = varianceSquared / count
    for i in range(rc):
        for j in range(rc):
            featureij = results[i,j,:]
            if iscity[i,j]:
                im[i*64:(i+1)*64,j*64:(j+1)*64,0] = 255
            if i != (rc-1):
                featureneig = results[i+1,j,:]
                EdgeStrength = np.exp(-(np.sum(np.square(featureij - featureneig))) / (2 * varianceSquared))
                # print(EdgeStrength)
                y = (i+1)*64
                x = j*64+32
                cv2.circle(im, (x, y), int(EdgeStrength*10)+2, (0, 0, 255), -1)  # 修改最后一个参数
                im[(i+1) * 64-1:(i + 1) * 64+1, j * 64:(j + 1) * 64, :] = 255
            if j != (rc-1):
                featureneig = results[i,j+1,:]
                EdgeStrength = np.exp(-(np.sum(np.square(featureij - featureneig))) / (2 * varianceSquared))
                # print(EdgeStrength)
                y = i * 64 + 32
                x = (j + 1) * 64
                cv2.circle(im, (x, y), int(EdgeStrength * 10)+2, (0, 0, 255), -1)  # 修改最后一个参数
                im[i * 64:(i + 1) * 64, (j + 1) * 64-1:(j + 1) * 64+1, :] = 255
    return im

if __name__ == "__main__":
    # modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark'  'Mynet']
    modelName = 'Mynet'
    modelDir = r'D:\MyProject\Tensorflow\workspace\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'
    imagePath = r'E:\DATA\GF2\性能测试\images\08\part1.tif'
    featuresDir = r'D:\MyProject\GraduationProject\DATA\features\similarity'
    global results
    # ------------------------------------------------------------------------------------------------------------------
    # 抽取特征
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
    # ------------------------------------------------------------------------------------------------------------------
    # 读取图像信息
    dataset = gdal.Open(imagePath)
    if dataset == None:
        print(imagePath + "掩膜失败，文件无法打开")
        dataEmpty = True
    # 获取图像信息
    srcWidth = dataset.RasterXSize  # 栅格矩阵的列数
    srcHeight = dataset.RasterYSize  # 栅格矩阵的行数
    data = dataset.ReadAsArray(xoff=0, yoff=0, xsize=srcWidth, ysize=srcHeight,
                               buf_xsize=srcWidth, buf_ysize=srcHeight, buf_type=gdal.GDT_UInt16)  # 获取数据
    data = unit_16_to_8(data)
    # cv2.imwrite(os.path.join(featuresDir,'im.bmp'),data)
    # ------------------------------------------------------------------------------------------------------------------
    # PCA降维
    # pca_res = PCA_im(results)
    # cv2.imwrite(os.path.join(featuresDir,'pca.bmp'),pca_res)
    # ------------------------------------------------------------------------------------------------------------------
    # 相似性可视化
    simi = block_similarity(data)
    cv2.imwrite(os.path.join(featuresDir,'simi.bmp'),simi)



