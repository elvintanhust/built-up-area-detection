import cv2
import numpy as np
import gdal


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

def Tif2BMP():
    in_path = r'E:\DATA\WordView\WorldView3\OR2A, Madrid, Spain, 30cm_053972858010\053972858010_01_P001_PSH\14SEP09105708-S2AS_R1C1-053972858010_01_P001.TIF'
    out_path = r'E:\DATA\WordView\WorldView3\OR2A, Madrid, Spain, 30cm_053972858010\053972858010_01_P001_PSH\Gray.bmp'
    dataset = gdal.Open(in_path)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    imR = im_data[2, :, :]
    imG = im_data[1, :, :]
    imB = im_data[0, :, :]
    imR = unit_16_to_8(imR)
    imG = unit_16_to_8(imG)
    imB = unit_16_to_8(imB)
    im = np.zeros([im_width, im_height, 3])
    im[:, :, 0] = imR
    im[:, :, 1] = imG
    im[:, :, 2] = imB
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_path, im, )

def ReduceResolution():
    in_path = r'E:\DATA\WordView\WorldView3\OR2A, Madrid, Spain, 30cm_053972858010\053972858010_01_P001_PSH\Gray.bmp'
    out_path = r'E:\DATA\WordView\WorldView3\OR2A, Madrid, Spain, 30cm_053972858010\053972858010_01_P001_PSH\Gray_LR.bmp'
    im = cv2.imread(in_path)
    im = cv2.resize(im,(6400,6400))
    cv2.imwrite(out_path, im, )

if __name__ == "__main__":
    ReduceResolution()