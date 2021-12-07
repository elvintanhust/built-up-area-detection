"""
Implementation of t-SNE based on Van Der Maaten and Hinton (2008)
http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tsne.load_data import load_mnist
from tsne.tsne import estimate_sne, tsne_grad, symmetric_sne_grad, q_tsne, q_joint
from tsne.tsne import p_joint
# ----------------------------------------------------------------------------------------------------------------------
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
from Function.TFRecord2image import next_city_batch
from Function.TFRecord2image import next_noncity_batch

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

imSize = 64
batch_size = 50


FEATURESIZE = 256
features = np.zeros([2000, FEATURESIZE], float)
labels = np.zeros([2000, 1], float)
dataEmpty = False
tifTime = 0

def visualization_batch(modelName,modelDir):
    global features
    global labels
    height, width = 64,64
    with tf.Session() as sess:
        batch_size = 10

        city_image_batch, city_label_batch = next_city_batch(model='test', num_epochs=1,
                                                             batch_size=batch_size, out_height=height,
                                                             out_width=width, n_classes=2,
                                                             isProcess=True)
        noncity_image_batch, noncity_label_batch = next_noncity_batch(model='test', num_epochs=1,
                                                                      batch_size=batch_size, out_height=height,
                                                                      out_width=width, n_classes=2,
                                                                      isProcess=True)
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
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
        # ------------------------------------------------------------------------------------------------------------------
        # 取某一层的输出为输出新建为model，采用函数模型
        predict_layer = Model(inputs=MODEL.model.input,
                              outputs=MODEL.model.get_layer('feature').output)
        k = 0
        for i in range(100):
            c_image_batch, c_label_batch = sess.run([city_image_batch, city_label_batch])
            feature = predict_layer.predict_on_batch(c_image_batch)
            for j in range(batch_size):
                features[k,:] = feature[j,:]
                labels[k] = 0
                k = k + 1
        for i in range(100):
            n_image_batch, n_label_batch = sess.run([noncity_image_batch, noncity_label_batch])
            feature = predict_layer.predict_on_batch(n_image_batch)
            for j in range(batch_size):
                features[k,:] = feature[j,:]
                labels[k] = 1
                k = k + 1

        coord.request_stop()
        coord.join()
# ----------------------------------------------------------------------------------------------------------------------
# Set global parameters
NUM_POINTS = 200            # Number of samples from MNIST
CLASSES_TO_USE = [0, 1, 8]  # MNIST classes to use
PERPLEXITY = 20  # 混淆度通常20-50
SEED = 1                    # Random seed
MOMENTUM = 0.9
LEARNING_RATE = 10.
NUM_ITERS = 800             # Num iterations to train for
TSNE = True                # If False, Symmetric SNE
NUM_PLOTS = 1               # Num. times to plot in training


def main():
    global features
    global labels

    # numpy RandomState for reproducibility
    rng = np.random.RandomState(SEED)

    # Load the first NUM_POINTS 0's, 1's and 8's from MNIST
    # X, y = load_mnist('datasets/',
    #                   digits_to_keep=CLASSES_TO_USE,
    #                   N=NUM_POINTS)

    # Obtain matrix of joint probabilities p_ij
    P = p_joint(features, PERPLEXITY)

    # Fit SNE or t-SNE
    Y = estimate_sne(features, labels, P, rng,
                     num_iters=NUM_ITERS,
                     q_fn=q_tsne if TSNE else q_joint,
                     grad_fn=tsne_grad if TSNE else symmetric_sne_grad,
                     learning_rate=LEARNING_RATE,
                     momentum=MOMENTUM,
                     plot=NUM_PLOTS)
    return Y

if __name__ == "__main__":
    modelName = 'Mynet'
    modelDir = r'D:\MyProject\Tensorflow\workspace\models\Mynet\MaxOut+PoolBranch+FC\2017-12-03_17-39-28-CE'
    visualization_batch(modelName, modelDir)

    Y = main()
    if os.path.exists(os.path.join(r'D:\matlab_test', 'SNE.txt')):
        os.remove(os.path.join(r'D:\matlab_test', 'SNE.txt'))
    f = open(os.path.join(r'D:\matlab_test', 'SNE.txt'), 'a+')
    for i in range(2000):
        f.write('%f %f\n'%(Y[i,0],Y[i,1]))
    f.close()
    print(Y)