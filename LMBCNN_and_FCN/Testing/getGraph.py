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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

blockLength = 4096
smallBlock = 64
imSize = 64
batch_size = 160
srcWidth = 10240
srcHeight = 10240

InputImageSize = 10240
zuobiao = np.zeros([batch_size, 2], int)
image_batch = np.zeros([batch_size, imSize, imSize, 1], int)
results = np.zeros([InputImageSize//imSize, InputImageSize//imSize], float)
im_predicts = np.zeros([InputImageSize//imSize, InputImageSize//imSize], float)

FEATURESIZE = 256
features = np.zeros([InputImageSize//imSize, InputImageSize//imSize, FEATURESIZE], float)
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
            # data = cv2.imread(imagePath)
            # data = data[:,:,1]
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
                    # flag_pixel_hist = True
                    # if x == 0 or y == 0 or x == (nX - 1) or y == (nY - 1):
                    #     hist_temp = cv2.calcHist([imagePart], [0], None, [256], [0.0, 255.0])
                    #     if np.sum(hist_temp > 5) < 10:
                    #         flag_pixel_hist = False
                    # ave = np.average(imagePart[:, :])
                    # if ave >= 20 and ave <= 230 and flag_pixel_hist:
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
                               outputs=MODEL.model.layers[55].output) # Activation：256

    # ------------------------------------------------------------------------------------------------------------------
    k = 1
    while not dataEmpty:
        dataSingal.wait()  # 等待数据准备
        imagesToClassify = image_batch
        zuobiaoTemp = zuobiao
        processSingal.set()  # 数据取完，可以继续准备数据
        cv2.waitKey(1)
        processSingal.clear()  # 下一批数据准备好后阻塞数据线程

        # for i in range(batch_size):
        #     im = imagesToClassify[i,:,:,:]
        #     im = im.astype(np.uint8)
        #     cv2.imshow('show',im)
        #     cv2.waitKey(1000)

        imagesToClassify = imagesToClassify.astype(np.float32)
        imagesToClassify = imagesToClassify / 255
        # show_batch('show', imagesToClassify, showlabel=False, col=8, row=8, channel=1, predicts=None, labels=None)
        feature = predict_layer.predict_on_batch( imagesToClassify)
        predict = MODEL.model.predict_on_batch(imagesToClassify)
        print('处理第%d批数据\n' % k)
        k = k + 1
        batch_size_now = batch_size - np.sum(zuobiaoTemp[:, 0] == -1)

        for bb in range(batch_size_now):
            y = zuobiaoTemp[bb, 0]
            x = zuobiaoTemp[bb, 1]
            results[y, x] = predict[bb][0]
            features[y][x][:] = feature[bb][:]

def getFeaturesFromBMP(modelName,modelDir,imagePath):
    with tf.Session() as sess:
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 模型初始化
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
        features_layer = Model(inputs=MODEL.model.input,outputs=[MODEL.model.layers[55].output,MODEL.model.layers[57].output])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # ------------------------------------------------------------------------------------------------------------------
        # 测试
        im = cv2.imread(imagePath)
        [w,h,c] = im.shape
        im = im[:, :, 1]
        global features
        global results
        # 提取深度特征
        for i in range(w//64):
            city_image_batch = np.zeros([h//64, 64, 64, 1], int)
            for j in range(h//64):
                city_image_batch[j, :, :, 0] = im[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64]
            city_image_batch = city_image_batch.astype(np.float32)
            city_image_batch = city_image_batch / 255
            feature_temp = features_layer.predict_on_batch(city_image_batch)
            # predict_temp = MODEL.model.predict_on_batch(city_image_batch)
            features[i, :, :] = feature_temp[0][:, :]
            results[i, :] = feature_temp[1][:,0]
        # ------------------------------------------------------------------------------------------------------------------
        coord.request_stop()
        coord.join()

def writeResult(save_dir):
    global features
    global srcHeight
    global srcWidth
    global results
    nX_s = (srcWidth - 1) // smallBlock + 1  # 计算列方向上smallBlock大小块数
    nY_s = (srcHeight - 1) // smallBlock + 1  # 计算行方向smallBlock大小块数
    fg = np.zeros([nY_s, nX_s,3],np.uint8)
    bg = np.zeros([nY_s, nX_s, 3], np.uint8)
    # 保存类别概率
    print('保存类别概率')
    if os.path.exists(os.path.join(save_dir, 'Predicts.txt')):
        os.remove(os.path.join(save_dir, 'Predicts.txt'))
    f = open(os.path.join(save_dir, 'Predicts.txt'), 'a+')
    for i in range(nY_s):
        for j in range(nX_s):
            f.write('%d %d %f\n'%(i,j,results[i,j]))
    f.close()
    # 计算特征距离的均值
    print('计算特征距离的均值')
    varianceSquared = 0.0
    count = 0
    for i in range(nY_s):
        for j in range(nX_s):
            if results[i][j] > 0.5:
                fg[i][j][:] = 255
            elif results[i][j] <=0.5:
                bg[i][j][:] = 255
            currNodeId = i * nX_s + j
            currFeature = features[i][j][:]
            di = [-1, -1, 0, 1]
            dj = [0, 1, 1, 1]
            for si, sj in zip(di, dj):
                if not (si + i < nY_s and si + i >= 0 and sj + j < nX_s):
                    continue
                nFeature = features[si + i][sj + j][:]
                varianceSquared += np.sum(np.square(currFeature - nFeature))
                count += 1
    cv2.imwrite(os.path.join(save_dir,'fg.bmp'),fg)
    cv2.imwrite(os.path.join(save_dir, 'bg.bmp'), bg)
    varianceSquared = varianceSquared/count
    # 计算graph边的权重
    print('计算graph边的权重')
    if os.path.exists(os.path.join(save_dir, 'Edges.txt')):
        os.remove(os.path.join(save_dir, 'Edges.txt'))
    if os.path.exists(os.path.join(save_dir, 'Distant.txt')):
        os.remove(os.path.join(save_dir, 'Distant.txt'))
    if os.path.exists(os.path.join(save_dir, 'CosDis.txt')):
        os.remove(os.path.join(save_dir, 'CosDis.txt'))
    f = open(os.path.join(save_dir, 'Edges.txt'), 'a+')
    f_dis = open(os.path.join(save_dir, 'Distant.txt'), 'a+')
    f_cos_dis = open(os.path.join(save_dir, 'CosDis.txt'), 'a+')
    for i in range(nY_s):
        for j in range(nX_s):
            currNodeId = i * nX_s + j
            currFeature = features[i][j][:]
            di = [-1, -1, 0, 1]
            dj = [0, 1, 1, 1]
            for si, sj in zip(di, dj):
                if not (si + i < nY_s and si + i >= 0 and sj + j < nX_s):
                    continue
                nNodeID = (si + i) * nX_s + sj + j
                nFeature = features[si + i][sj + j][:]
                distance = np.sum(np.square(currFeature - nFeature))
                EdgeStrength = np.exp(-(distance)/(2*varianceSquared))

                cosDistance = (np.dot(currFeature,nFeature)/(np.linalg.norm(currFeature)*np.linalg.norm(nFeature)+0.000001)+1.0)/2.0
                # currDist = np.sqrt(si * si + sj * sj)
                # EdgeStrength = (0.95 * EdgeStrength +0.05) / currDist
                f.write('%d %d %f\n'%(currNodeId, nNodeID, EdgeStrength))
                f_dis.write('%d %d %f\n'%(currNodeId, nNodeID, distance))
                f_cos_dis.write('%d %d %f\n'%(currNodeId, nNodeID, cosDistance))
    f.close()
    f_dis.close()
    f_cos_dis.close()
    # 计算graph辅助节点的边
    print('计算graph辅助节点的边')
    if os.path.exists(os.path.join(save_dir, 'Auxiliary.txt')):
        os.remove(os.path.join(save_dir, 'Auxiliary.txt'))
    f = open(os.path.join(save_dir, 'Auxiliary.txt'), 'a+')
    # data = np.zeros([nY_s*nX_s, FEATURESIZE],float)
    # for i in range(nY_s):
    #     for j in range(nX_s):
    #         currNodeId = i * nX_s + j
    #         currFeature = features[i][j][:]
    #         data[currNodeId][:] = currFeature
    # print('开始聚类')
    # estimator = KMeans(n_clusters=64, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto' )
    # estimator.fit(data)  # 聚类
    # label_pred = estimator.labels_  # 获取聚类标签
    # centroids = estimator.cluster_centers_  # 获取聚类中心
    # print('结束聚类')
    # for i in range(nY_s):
    #     for j in range(nX_s):
    #         currNodeId = i * nX_s + j
    #         currFeature = features[i][j][:]
    #         label = label_pred[currNodeId]
    #         nFeature = centroids[label]
    #         EdgeStrength = np.exp(-(np.sum(np.square(currFeature - nFeature))) / (2 * varianceSquared))
    #         f.write('%d %d %f\n' % (currNodeId, label, EdgeStrength))
    for i in range(nY_s):
        for j in range(nX_s):
            currNodeId = i * nX_s + j
            bin = int(results[i][j]*1024)
            f.write('%d %d\n' % (currNodeId, bin))
    f.close()
    return 0

if __name__ == "__main__":
    #-------------------------------------------------------------------------------------------------------------------------------------
    # 处理10240*10240图像
    # modelName = ['Mobile'  'Alexnet'  'Inception'  'Tinydark'  'Mynet']
    modelName = 'Mynet'
    modelDir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'

    imInd = '110'
    imagePath = os.path.join(r'E:\DATA\GF2\PerformanceTest\images', imInd)
    # imagePath = r'E:\DATA\quickbird\QB_11Nov2007\425665580\07NOV11192134-P2AS_R1C1-052623283010_01_P001.TIF'
    print("start\n")
    start_time = time.time()
    if 0:
        imagePath = os.path.join(imagePath, 'part1.tif')
        dataSingal = threading.Event()
        processSingal = threading.Event()
        tIM = threading.Thread(target=getImageBatch, args=(dataSingal, processSingal,imagePath))
        tCL = threading.Thread(target=getFeatures, args=(dataSingal, processSingal,modelName,modelDir))
        tIM.start()
        tCL.start()
        tIM.join()
        tCL.join()
    else:
        imagePath = os.path.join(imagePath, 'part1.bmp')
        getFeaturesFromBMP(modelName,modelDir, imagePath)
    save_dir =os.path.join( r'E:\DATA\GF2\PerformanceTest\LMB-CNN',imInd)
    save_dir = os.path.join(save_dir,'GraphCut')
    # save_dir = r'E:\DATA\GF2\PerformanceTest\MyNet\QuickBird\GraphCut'
    # writeResult(save_dir = save_dir)

    times = time.time() - start_time
    print('总计耗时：%.6f\n' % (times,))
    # -------------------------------------------------------------------------------------------------------------------------------------
    # 处理2048*2048图像
    # modelName = 'Mynet'
    # modelDir = r'D:\MyProject\GraduationProject\TensorFlow\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'
    #
    # imInd = '110'
    # imagePath = os.path.join(r'E:\DATA\GF2\PerformanceTest\smallImages', imInd)
    # print("start\n")
    # start_time = time.time()
    # imagePath = imagePath+ '.bmp'
    # getFeaturesFromBMP(modelName, modelDir, imagePath)
    # save_dir = os.path.join(r'E:\DATA\GF2\PerformanceTest\SuperviseAlgorithm\MLB-CNN+GC', imInd)
    # # save_dir = r'E:\DATA\GF2\PerformanceTest\MyNet\QuickBird\GraphCut'
    # writeResult(save_dir=save_dir)
    #
    # times = time.time() - start_time
    # print('总计耗时：%.6f\n' % (times,))
