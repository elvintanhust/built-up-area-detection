import gdal
import numpy as np
import cv2
import os

# 将16位数据线性裁剪拉伸并转为8位
def unit_16_to_8_mul(data):
    channel,height, width = data.shape
    bandMin = np.zeros([channel,1])
    bandMax = np.zeros([channel, 1])

    for i in range(channel):
        pHistgoram = cv2.calcHist([data[i,:,:]], [0], None, [65535], [0.0, 65535.0])
        sumPix = width * height - pHistgoram[0]
        weight_0 = 0.001
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
            if (pDataMax - pDataMin > 400) and (weight_0 < 0.04):
                weight_0 = weight_0 + 0.001
            else:
                break
        while 1:
            if pDataMax - pDataMin < 100:
                pDataMax = pDataMax + 50
                pDataMin = pDataMin - 50
            else:
                break
        bandMin[i] = pDataMin
        bandMax[i] = pDataMax
    for i in range(channel):
        res = data[i,:,:]
        res[res > bandMax[i]] = bandMax[i]
        res[res < bandMin[i]] = bandMin[i]
        data[i,:,:] = res
    res = np.zeros([height, width,3], float)
    for i in [1,2,3]:
        res[:,:,i-1] = (data[i,:,:]-bandMin[i])/(bandMax[i]-bandMin[i]+0.0)*255
    res = res.astype(np.uint8)
    res = cv2.resize(res,(2560,2560))
    return res

def getMulImage(rootDir):
    panFilePath = os.path.join(rootDir ,'part1.tif')
    mulFilePath = os.path.join(rootDir ,'ortho_m.tif')

    panFile = gdal.Open(panFilePath)
    if panFile == None:
        print(rootDir + "掩膜失败，文件无法打开")
        return
    mulFile = gdal.Open(mulFilePath)
    if mulFile == None:
        print(rootDir + "掩膜失败，文件无法打开")
        return
    # 获取全色图像信息
    panWidth = panFile.RasterXSize  # 栅格矩阵的列数
    panHeight = panFile.RasterYSize  # 栅格矩阵的行数
    panBands = panFile.RasterCount  # 波段数
    panTrans = panFile.GetGeoTransform()  # 获取仿射矩阵信息
    panProj = panFile.GetProjection()  # 获取投影信息

    # 获取多光谱图像信息
    mulWidth = mulFile.RasterXSize  # 栅格矩阵的列数
    mulHeight = mulFile.RasterYSize  # 栅格矩阵的行数
    mulBands = mulFile.RasterCount  # 波段数
    mulTrans = mulFile.GetGeoTransform()  # 获取仿射矩阵信息
    mulProj = mulFile.GetProjection()  # 获取投影信息

    geo_mStartX = mulTrans[0] + 0 * mulTrans[1] + 0 * mulTrans[2]
    geo_mStartY = mulTrans[3] + 0 * mulTrans[4] + 0 * mulTrans[5]
    geo_mEndX = mulTrans[0] + (mulWidth - 1) * mulTrans[1] + (mulHeight - 1) * mulTrans[2]
    geo_mEndY = mulTrans[3] + (mulWidth - 1) * mulTrans[4] + (mulHeight - 1) * mulTrans[5]

    geo_mStepX = (geo_mEndX - geo_mStartX) / mulWidth
    geo_mStepY = (geo_mEndY - geo_mStartY) / mulHeight

    # 四角点对应多光谱图像上坐标
    p1x = 0
    p1y = 0
    p2x = panWidth
    p2y = panHeight

    Xgeo1 = panTrans[0] + p1x * panTrans[1] + p1y * panTrans[2]
    Ygeo1 = panTrans[3] + p1x * panTrans[4] + p1y * panTrans[5]
    Xgeo2 = panTrans[0] + p2x * panTrans[1] + p2y * panTrans[2]
    Ygeo2 = panTrans[3] + p2x * panTrans[4] + p2y * panTrans[5]

    if ((Xgeo1 - geo_mStartX) / (geo_mStepX+0.0) - int((Xgeo1 - geo_mStartX) /  (geo_mStepX+0.0)) < 0.5):
        Xindex1 = int((Xgeo1 - geo_mStartX) / (geo_mStepX+0.0))
    else:
        Xindex1 = int((Xgeo1 - geo_mStartX) / (geo_mStepX+0.0)) + 1

    if ((Ygeo1 - geo_mStartY) / (geo_mStepY+0.0) - int((Ygeo1 - geo_mStartY) / (geo_mStepY+0.0)) < 0.5):
        Yindex1 = int((Ygeo1 - geo_mStartY) / (geo_mStepY+0.0))
    else:
        Yindex1 = int((Ygeo1 - geo_mStartY) / (geo_mStepY+0.0)) + 1

    if ((Xgeo2 - geo_mStartX) /  (geo_mStepX+0.0) - int((Xgeo2 - geo_mStartX) /  (geo_mStepX+0.0)) < 0.5):
        Xindex2 = int((Xgeo2 - geo_mStartX) / (geo_mStepX+0.0))
    else:
        Xindex2 = int((Xgeo2 - geo_mStartX) / (geo_mStepX+0.0)) + 1

    if ((Ygeo2 - geo_mStartY) / (geo_mStepY+0.0) - int((Ygeo2 - geo_mStartY) / (geo_mStepY+0.0)) < 0.5):
        Yindex2 = int((Ygeo2 - geo_mStartY) / (geo_mStepY+0.0))
    else:
        Yindex2 = int((Ygeo2 - geo_mStartY) / (geo_mStepY+0.0)) + 1


    tmpHeight = Yindex2 - Yindex1 + 1
    tmpWidth = Xindex2 - Xindex1 + 1
    # 读取多光谱数据
    mulData = mulFile.ReadAsArray(xoff=Xindex1, yoff=Yindex1, xsize=tmpWidth, ysize=tmpHeight,
                               buf_xsize=tmpWidth, buf_ysize=tmpHeight, buf_type=gdal.GDT_UInt16)
    mulData = unit_16_to_8_mul(mulData)
    cv2.imwrite(os.path.join(rootDir ,'ortho_m.bmp'),mulData)

def getMulImage(path, name = None):

    mulFile = gdal.Open(path)
    if mulFile == None:
        print(path + "掩膜失败，文件无法打开")
        return

    # 获取多光谱图像信息
    mulWidth = mulFile.RasterXSize  # 栅格矩阵的列数
    mulHeight = mulFile.RasterYSize  # 栅格矩阵的行数

    # 读取多光谱数据
    mulData = mulFile.ReadAsArray(xoff=0, yoff=0, xsize=mulWidth, ysize=mulHeight,
                               buf_xsize=mulWidth, buf_ysize=mulHeight, buf_type=gdal.GDT_UInt16)
    mulData = unit_16_to_8_mul(mulData)
    cv2.imwrite(r'E:\DATA\quickbird\testData\ortho_m.bmp',mulData)

if __name__ == "__main__":
    # inds = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
    #         '101', '102', '103', '104', '105', '106', '107', '108', '109', '110']
    # rootDir = r'E:\DATA\GF2\PerformanceTest\images'
    # for i in inds:
    #     getMulImage(os.path.join(rootDir,i))
    getMulImage(r"E:\DATA\quickbird\QB_11Nov20007\052623283010_01_P001_MUL\07NOV11192134-M2AS_R1C1-052623283010_01_P001.TIF")