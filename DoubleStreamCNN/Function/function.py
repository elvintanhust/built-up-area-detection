import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import random
import time
import tensorflow as tf
import time

def plot_history(history, saver_dir):
    """
    绘制曲线
    :param history: MODEL.model.fit_generator()返回值
    :param saver_dir: 曲线保存路径
    :return:
    """
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'], 'b')
    plt.plot(history.history['val_acc'], 'y')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(saver_dir, 'accuracy.jpeg'))
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'], 'b')
    plt.plot(history.history['val_loss'], 'y')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(saver_dir, 'loss.jpeg'))
    plt.close()

def plot_curv(train_history, val_history, saver_dir):
    """
    绘制曲线
    :param train_history: 训练集loss和acc记录
    :param val_history: 验证集loss和acc记录
    :param saver_dir: 曲线保存路径
    :return:
    """
    len_t = train_history.__len__()
    len_v = val_history.__len__()

    train_acc,train_loss = [],[]
    for i in range(len_t):
        train_acc.append(train_history[i][1])
        train_loss.append(train_history[i][0])
    val_acc_c,val_acc_nc,val_loss_c,val_loss_nc = [],[],[],[]
    for i in range(len_v):
        val_acc_c.append(val_history[i][1])
        val_acc_nc.append(val_history[i][3])
        val_loss_c.append(val_history[i][0])
        val_loss_nc.append(val_history[i][2])
    val_acc = []
    for i, j in zip(val_acc_c, val_acc_nc):
        summ = (i + j)/2
        val_acc.append(summ)
    val_loss = []
    for i, j in zip(val_loss_c, val_loss_nc):
        summ = (i + j) / 2
        val_loss.append(summ)

    # summarize history for accuracy
    plt.plot(train_acc, 'b')
    plt.plot(val_acc, 'y')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(saver_dir, 'accuracy.jpeg'))
    plt.close()
    # summarize history for loss
    plt.plot(train_loss, 'b')
    plt.plot(val_loss, 'y')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(saver_dir, 'loss.jpeg'))
    plt.close()
    # summarize history for val accuracy
    plt.plot(val_acc_c, 'b')
    plt.plot(val_acc_nc, 'y')
    plt.title('two class val accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['city', 'noncity'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(saver_dir, 'val_accuracy.jpeg'))
    plt.close()
    # summarize history for val loss
    plt.plot(val_loss_c, 'b')
    plt.plot(val_loss_nc, 'y')
    plt.title('two class val accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['city', 'noncity'], loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(saver_dir, 'val_loss.jpeg'))
    plt.close()

def show_batch(name,images,showlabel = False, col = 10, row = 10, height = 64, width = 64, channel = 3,predicts = None, labels = None):
    """
    显示一个batch的图像
    :param name: 窗口名，字符串
    :param images: 一个batch的图像
    :param showlabel: 是否显示图像标签，若是则需给出predicts和labels
    :param col: 显示图像列数
    :param row: 显示图像行数
    :param channel: 图像通道数，单通道或三通道
    :param predicts: 图像预测值
    :param labels: 图像真值
    :return:
    """
    if channel == 3:
        ims = np.zeros([height*col, width*row,3],np.uint8)
    elif channel == 1:
        ims = np.zeros([height * col, width * row], np.uint8)
    k = 0
    if showlabel:
        # 创建一个矩形，来让我们在图片上写文字，参数依次定义了文字类型，高，宽，字体厚度等。。
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        # 将文字框加入到图片中，(5,20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色
        for i in range(col):
            for j in range(row):
                if channel == 3:
                    im = images[k,:,:,:]
                    im = im * 255
                    im = im.astype(np.uint8)
                    im = cv2.resize(im, (height, width))
                elif channel == 1:
                    im = images[k, :, :, :]
                    im = im * 255
                    im = im.astype(np.uint8)
                    im = cv2.resize(im, (height, width))
                if predicts[k] == 0:
                    cv2.putText(im, "city", (5, 40), font,0.5, (255, 0, 0),1)
                else:
                    cv2.putText(im, "non", (5, 40), font, 0.5, (255, 0, 0),1)
                if labels[k] == 0:
                    cv2.putText(im, "city", (5, 10), font, 0.5, (255, 0, 0),1)
                else:
                    cv2.putText(im, "non", (5, 10), font, 0.5, (255, 0, 0),1)
                if predicts[k] != labels[k]:
                    im[0:20, 45:64, 0] = 255
                    im[20:40, 45:64, 1] = 255
                    im[40:64, 45:64, 2] = 255
                if channel == 3:
                    ims[(i*height):(i*height + height),(j*width):(j*width + width),:] = im
                elif channel == 1:\
                    ims[(i * height):(i * height + height), (j * width):(j * width + width)] = im
                k = k + 1
    else:
        for i in range(col):
            for j in range(row):
                if channel == 3:
                    im = images[k, :, :, :]
                    im = im * 255
                    im = im.astype(np.uint8)
                    im = cv2.resize(im, (height, width))
                    ims[(i * height):(i * height + height), (j * width):(j * width + width), :] = im
                elif channel == 1:
                    im = images[k, :, :, :]
                    im = im * 255
                    im = im.astype(np.uint8)
                    im = cv2.resize(im, (height, width))
                    ims[(i * height):(i * height + height), (j * width):(j * width + width)] = im
                k = k + 1
    cv2.imshow(name,ims)
    cv2.waitKey(3000)
    # cv2.destroyWindow(name)
    return ims

def add_noise(image):
    """
    为图像添加均值0方差1的高斯噪声
    :param image: 输入单通道图像
    :return:
    """
    mu, sigma = 0, 1
    height, width = image.shape[0],image.shape[1]
    noise = np.random.normal(mu, sigma, [height, width])
    noise[noise > (3 * sigma)] = (3 * sigma)
    noise[noise < (-3 * sigma)] = (-3 * sigma)
    noise = noise.astype(int)
    image = image + noise
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)
    return image


def image_preprocess(image = None,height = 64, width = 64):
    """
    图像预处理
    :param image: 输入单通道图像
    :param height: 输出图像高
    :param width: 输出图像宽
    :return:
    """
    zoom_range = 0.1

    if len(image.shape) == 3:
        h, w, c = image.shape
        center = (w / 2, h / 2)
        angle = random.randint(0,4)*90
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        h_r = random.randint(0, int(h * zoom_range))
        w_r = random.randint(0, int(w * zoom_range))
        image = image[h_r:h-h_r, w_r:w-w_r ,:]

        image = cv2.resize(image, (height, width))
        image = add_noise(image[:,:,0])
    elif len(image.shape) == 2:
        h, w = image.shape
        center = (w / 2, h / 2)
        angle = random.randint(0, 4) * 90
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        h_r = random.randint(0, int(h * zoom_range))
        w_r = random.randint(0, int(w * zoom_range))
        image = image[h_r:h - h_r, w_r:w - w_r]

        image = cv2.resize(image, (height, width))
        image = add_noise(image)

    return image
    # cv2.imshow("show", image)
    # cv2.waitKey(1000)
    # cv2.destroyWindow("show")

# start = time.time()
# for i in range(64*1000):
#     image_preprocess()
# end = time.time()
# print("%d\n" % int(end-start))

class getDataByList:
    """
    获取batch数据
    __init__(self,model = 'test',batch_size = 100,height = 128,width = 128)
    初始化batch_size，height，width，数据路径也在该函数中指定
    get_city_batch(self, singleloop = True, preprocess = True)
    获取一个batch的city数据，singleloop指定是否每个样本只读一次，preprocess指定是否预处理图像
    get_noncity_batch(self, singleloop = True, preprocess = True)
    获取一个batch的city数据，singleloop指定是否每个样本只读一次，preprocess指定是否预处理图像
    get_next_batch(self, singleloop = True, preprocess = True)
    获取一个batch的数据，前半部分city，后半部分noncity，singleloop指定是否每个样本只读一次，preprocess指定是否预处理图像
    """
    def __init__(self,model = 'test',batch_size = 100,panheight = 64,panwidth = 64, mulheight = 16,mulwidth = 16):
        self.model = model
        self.batch_size = batch_size
        self.panheight = panheight
        self.panwidth = panwidth
        self.mulheight = mulheight
        self.mulwidth = mulwidth
        self.cityInd = 0
        self.noncityInd = 0
        self.pan_path = r'E:\DATA\GF2\panchromatic64'
        self.mul_path = r'E:\DATA\GF2\multispectral'
        self.pan_city_dir = os.path.join(self.pan_path,model,'city')
        self.pan_noncity_dir = os.path.join(self.pan_path, model, 'noncity')
        self.cityFiles = os.listdir(self.pan_city_dir)
        self.noncityFiles = os.listdir(self.pan_noncity_dir)
        self.cityNum = self.cityFiles.__len__()
        self.noncityNum = self.noncityFiles.__len__()
        random.shuffle(self.cityFiles)
        random.shuffle(self.noncityFiles)
        self.MulCityImages = []
        self.MulNoncityImages = []
        for i in range(self.cityNum):
            dir = self.cityFiles[i]
            mul_im = cv2.imread(os.path.join(self.mul_path, dir))
            mul_im = cv2.resize(mul_im, (self.mulheight, self.mulwidth))
            self.MulCityImages.append(mul_im)
            print('读入多光谱City图像：%f' %(i/(self.cityNum+0.0),))
        for i in range(self.noncityNum):
            dir = self.noncityFiles[i]
            mul_im = cv2.imread(os.path.join(self.mul_path, dir))
            mul_im = cv2.resize(mul_im, (self.mulheight, self.mulwidth))
            self.MulNoncityImages.append(mul_im)
            print('读入多光谱Noncity图像：%f' % (i / (self.noncityNum + 0.0),))

    def get_city_batch(self, singleloop = True, preprocess = True):
        if self.cityInd + self.batch_size > self.cityNum:
            self.cityInd = 0
            random.shuffle(self.cityFiles)
            if singleloop:
                return [],[],True
        imagesToClassifyPan = np.zeros([self.batch_size, self.panheight, self.panwidth, 1])
        imagesToClassifyMul = np.zeros([self.batch_size, self.mulheight, self.mulwidth, 3])
        for i in range(self.batch_size):
            dir = self.cityFiles[self.cityInd]
            self.cityInd = self.cityInd + 1
            pan_im = cv2.imread(os.path.join(self.pan_city_dir, dir))
            mul_im = self.MulCityImages[self.cityInd-1]
            if preprocess:
                pan_im = image_preprocess(image = pan_im,height = self.panheight, width = self.panwidth)
                imagesToClassifyPan[i, :, :, :] = pan_im.reshape([self.panheight, self.panwidth, 1])

                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])
            else:
                pan_im = cv2.resize(pan_im, (self.panheight, self.panwidth))
                imagesToClassifyPan[i, :, :, :] = pan_im[:,:,1].reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])

            imagesToClassifyPan = imagesToClassifyPan.astype(np.float32)
            imagesToClassifyPan = imagesToClassifyPan / 255
            imagesToClassifyMul = imagesToClassifyMul.astype(np.float32)
            imagesToClassifyMul = imagesToClassifyMul / 255
        return imagesToClassifyPan,imagesToClassifyMul,False

    def get_noncity_batch(self, singleloop = True, preprocess = True):
        if self.noncityInd + self.batch_size > self.noncityNum:
            self.noncityInd = 0
            random.shuffle(self.noncityFiles)
            if singleloop:
                return [],[],True

        imagesToClassifyPan = np.zeros([self.batch_size, self.panheight, self.panwidth, 1])
        imagesToClassifyMul = np.zeros([self.batch_size, self.mulheight, self.mulwidth, 3])
        for i in range(self.batch_size):
            dir = self.noncityFiles[self.noncityInd]
            self.noncityInd = self.noncityInd + 1
            pan_im = cv2.imread(os.path.join(self.pan_noncity_dir, dir))
            mul_im = self.MulNoncityImages[self.noncityInd - 1]
            if preprocess:
                pan_im = image_preprocess(image=pan_im, height=self.panheight, width=self.panwidth)
                imagesToClassifyPan[i, :, :, :] = pan_im.reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])
            else:
                pan_im = cv2.resize(pan_im, (self.panheight, self.panwidth))
                imagesToClassifyPan[i, :, :, :] = pan_im[:, :, 1].reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])

        imagesToClassifyPan = imagesToClassifyPan.astype(np.float32)
        imagesToClassifyPan = imagesToClassifyPan / 255
        imagesToClassifyMul = imagesToClassifyMul.astype(np.float32)
        imagesToClassifyMul = imagesToClassifyMul / 255
        return imagesToClassifyPan, imagesToClassifyMul, False

    def get_next_batch(self, singleloop = True, preprocess = True):
        if self.cityInd + self.batch_size > self.cityNum:
            self.cityInd = 0
            random.shuffle(self.cityFiles)
            if singleloop:
                return [],[],True
        if self.noncityInd + self.batch_size//2 > self.noncityNum:
            self.noncityInd = 0
            random.shuffle(self.noncityFiles)
            if singleloop:
                return [],[],True

        imagesToClassifyPan = np.zeros([self.batch_size, self.panheight, self.panwidth, 1])
        imagesToClassifyMul = np.zeros([self.batch_size, self.mulheight, self.mulwidth, 3])
        for i in range(self.batch_size//2):
            dir = self.cityFiles[self.cityInd]
            self.cityInd = self.cityInd + 1
            pan_im = cv2.imread(os.path.join(self.pan_city_dir, dir))
            mul_im = self.MulCityImages[self.cityInd - 1]
            if preprocess:
                pan_im = image_preprocess(image=pan_im, height=self.panheight, width=self.panwidth)
                imagesToClassifyPan[i, :, :, :] = pan_im.reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])
            else:
                pan_im = cv2.resize(pan_im, (self.panheight, self.panwidth))
                imagesToClassifyPan[i, :, :, :] = pan_im[:, :, 1].reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])
        for i in range(self.batch_size//2,self.batch_size):
            dir = self.noncityFiles[self.noncityInd]
            self.noncityInd = self.noncityInd + 1
            pan_im = cv2.imread(os.path.join(self.pan_noncity_dir, dir))
            mul_im = self.MulNoncityImages[self.noncityInd - 1]
            if preprocess:
                pan_im = image_preprocess(image=pan_im, height=self.panheight, width=self.panwidth)
                imagesToClassifyPan[i, :, :, :] = pan_im.reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])
            else:
                pan_im = cv2.resize(pan_im, (self.panheight, self.panwidth))
                imagesToClassifyPan[i, :, :, :] = pan_im[:, :, 1].reshape([self.panheight, self.panwidth, 1])
                imagesToClassifyMul[i, :, :, :] = mul_im.reshape([self.mulheight, self.mulwidth, 3])

        imagesToClassifyPan = imagesToClassifyPan.astype(np.float32)
        imagesToClassifyPan = imagesToClassifyPan / 255
        imagesToClassifyMul = imagesToClassifyMul.astype(np.float32)
        imagesToClassifyMul = imagesToClassifyMul / 255
        return imagesToClassifyPan, imagesToClassifyMul, False


