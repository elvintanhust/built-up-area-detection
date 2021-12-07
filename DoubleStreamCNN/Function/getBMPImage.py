import os
import gdal
import numpy as np
import cv2
blockLength = 2048
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


def getBMPImage(ind):
    imagePath = os.path.join(r'E:\DATA\GF2\PerformanceTest\images',ind,'part1.tif')
    dataset = gdal.Open(imagePath)
    if dataset == None:
        print(imagePath + "掩膜失败，文件无法打开")
        return
    # 获取图像信息
    srcWidth = dataset.RasterXSize  # 栅格矩阵的列数
    srcHeight = dataset.RasterYSize  # 栅格矩阵的行数

    nX = (srcWidth - 1) // blockLength + 1  # 计算列方向上blockLength大小块数
    nY = (srcHeight - 1) // blockLength + 1  # 计算行方向blockLength大小块数

    # driver = gdal.GetDriverByName("GTiff")
    # dataout = driver.Create(os.path.join(os.path.dirname(imagePath),'result.tif'), srcWidth, srcHeight, 3, gdal.GDT_Byte)
    outImg = np.zeros([srcHeight,srcWidth])
    for x in range(nX):
        for y in range(nY):
            data = dataset.ReadAsArray(xoff=x * blockLength, yoff=y * blockLength, xsize=blockLength, ysize=blockLength,
                                       buf_xsize=blockLength, buf_ysize=blockLength, buf_type=gdal.GDT_UInt16)  # 获取数据
            data = unit_16_to_8(data)
            outImg[ y* blockLength:(y+1) * blockLength,x* blockLength:(x+1) * blockLength] = data
            # for i in range(3):
            #     dataout.GetRasterBand(i + 1).WriteArray(data[:,:,i],xoff=x * blockLength, yoff=y * blockLength)
    outPath = os.path.join(r'E:\DATA\GF2\PerformanceTest\images', ind, 'part1.bmp')
    cv2.imwrite(outPath,outImg)

def getBMPImage(path,name = None):

    dataset = gdal.Open(path)
    if dataset == None:
        print(path + "掩膜失败，文件无法打开")
        return
    # 获取图像信息
    srcWidth = dataset.RasterXSize  # 栅格矩阵的列数
    srcHeight = dataset.RasterYSize  # 栅格矩阵的行数

    nX = (srcWidth - 1) // blockLength + 1  # 计算列方向上blockLength大小块数
    nY = (srcHeight - 1) // blockLength + 1  # 计算行方向blockLength大小块数

    # driver = gdal.GetDriverByName("GTiff")
    # dataout = driver.Create(os.path.join(os.path.dirname(imagePath),'result.tif'), srcWidth, srcHeight, 3, gdal.GDT_Byte)
    outImg = np.zeros([srcHeight,srcWidth])
    for x in range(nX):
        for y in range(nY):
            data = dataset.ReadAsArray(xoff=x * blockLength, yoff=y * blockLength, xsize=blockLength, ysize=blockLength,
                                       buf_xsize=blockLength, buf_ysize=blockLength, buf_type=gdal.GDT_UInt16)  # 获取数据
            data = unit_16_to_8(data)
            outImg[ y* blockLength:(y+1) * blockLength,x* blockLength:(x+1) * blockLength] = data
            # for i in range(3):
            #     dataout.GetRasterBand(i + 1).WriteArray(data[:,:,i],xoff=x * blockLength, yoff=y * blockLength)
    outPath = os.path.join(r"E:\DATA\GF2\PAN_10240_10240\images\64.bmp")
    cv2.imwrite(outPath,outImg)

def getBMPImage_4channels(path,name = None):

    dataset = gdal.Open(path)
    if dataset == None:
        print(path + "掩膜失败，文件无法打开")
        return
    # 获取图像信息
    srcWidth = dataset.RasterXSize  # 栅格矩阵的列数
    srcHeight = dataset.RasterYSize  # 栅格矩阵的行数

    nX = (srcWidth - 1) // blockLength + 1  # 计算列方向上blockLength大小块数
    nY = (srcHeight - 1) // blockLength + 1  # 计算行方向blockLength大小块数

    # driver = gdal.GetDriverByName("GTiff")
    # dataout = driver.Create(os.path.join(os.path.dirname(imagePath),'result.tif'), srcWidth, srcHeight, 3, gdal.GDT_Byte)
    outImg = np.zeros([srcHeight,srcWidth,3],np.uint8)


    for x in range(nX):
        for y in range(nY):
            data = dataset.ReadAsArray(xoff=x * blockLength, yoff=y * blockLength, xsize=blockLength, ysize=blockLength,
                                       buf_xsize=blockLength, buf_ysize=blockLength, buf_type=gdal.GDT_UInt16)  # 获取数据
            nr = data[0, :, :]
            nr.shape = [blockLength,blockLength]
            nr = unit_16_to_8(nr)
            outImg[ y* blockLength:(y+1) * blockLength,x* blockLength:(x+1) * blockLength, 0] = nr
            r = data[1, :, :]
            r.shape = [blockLength, blockLength]
            r = unit_16_to_8(r)
            outImg[y * blockLength:(y + 1) * blockLength, x * blockLength:(x + 1) * blockLength, 1] = r
            g = data[2, :, :]
            g.shape = [blockLength, blockLength]
            g = unit_16_to_8(g)
            outImg[y * blockLength:(y + 1) * blockLength, x * blockLength:(x + 1) * blockLength, 2] = g

            # for i in range(3):
            #     dataout.GetRasterBand(i + 1).WriteArray(data[:,:,i],xoff=x * blockLength, yoff=y * blockLength)
    outPath = os.path.join(r"D:\Pictures\sp\109.bmp")
    outImg = outImg.astype(np.uint8)
    cv2.imwrite(outPath,outImg)

if __name__ == '__main__':
    # inds = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','101','102','103','104','105','106','107','108','109','110']
    # for ind in inds:
    #     getBMPImage(ind)
    getBMPImage_4channels(r"D:\Pictures\sp\sp_file_109.tif")