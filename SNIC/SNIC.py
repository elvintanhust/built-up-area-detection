import cv2
import numpy as np
import queue
import math
from skimage import color
import time


def rgb2lab(im, channal):
    if channal == 1:
        h, w = im.shape
        lab = np.array([h, w, 3])
        lab[:, :, 0] = im
        lab[:, :, 1] = im
        lab[:, :, 2] = im
    elif channal == 3:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    return lab


def FindSeeds(width, height, numk):
    sz = width * height
    gridstep = int(math.sqrt(sz / numk) + 0.5)
    halfstep = int(gridstep / 2)
    h = height
    w = width
    xsteps = int(width / gridstep)
    ysteps = int(height / gridstep)
    err1 = abs(xsteps * ysteps - numk)
    err2 = abs(int(width / (gridstep - 1)) * int(height / (gridstep - 1)) - numk)
    if err2 < err1:
        gridstep -= 1.0
        xsteps = width // (gridstep)
        ysteps = height // (gridstep)
    numk = (xsteps * ysteps)
    kx = np.zeros([numk], int)
    ky = np.zeros([numk], int)
    y, rowstep, n = halfstep, 0, 0
    while y < height and n < numk:
        x = halfstep
        while x < width and n < numk:
            if y <= h - halfstep and x <= w - halfstep:
                kx[n] = x
                ky[n] = y
                n += 1
            x += gridstep
        y += gridstep
        rowstep += 1
    return kx, ky, numk


class NODE(object):
    def __init__(self, i=0, k=0, d=0):
        self.i = i  # the x and y values packed into one
        self.k = k  # the label
        self.d = d  # the distance

    # 重写小于号，比较函数
    def __lt__(self, other):  # operator <
        return self.d < other.d

    def __str__(self):
        return '(' + str(self.d) + ',' + str(self.i) + ',' + str(self.k) + ')'


def SNIC(lv, av, bv, width, height, innumk, compactness):
    """
    "Superpixels and Polygons using Simple Non-Iterative Clustering"
    Radhakrishna Achanta and Sabine Susstrunk
    CVPR 2017
    :param lv:
    :param av:
    :param bv:
    :param width:
    :param height:
    :param innumk: 预计超像素数目
    :param compactness:Compactness factor [10, 40]
    :return:
        labels:The labels of the segmented image
        outnumk:The number of labels
    """
    start = time.time()
    labels = np.zeros([width * height], int)
    labels[:] = -1
    w = width
    h = height
    sz = w * h
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]  # for 4 or 8 connectivity
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]  # for 4 or 8 connectivity
    dn8 = [-1, -w, 1, w, -1 - w, 1 - w, 1 + w, -1 + w]

    # Find seeds
    numk = innumk
    cx, cy, numk = FindSeeds(width, height, numk)  # the function may modify numk from its initial value
    print('findseed: %f' % (time.time() - start))
    start = time.time()
    # Initialize
    que = queue.PriorityQueue()
    for k in range(numk):
        temp_i = int(int(cx[k]) << 16 | int(cy[k]))
        temp_k = k
        temp_d = 0
        tempnode = NODE(temp_i, temp_k, temp_d)
        que.put(tempnode)

    kl = np.zeros([numk])
    ka = np.zeros([numk])
    kb = np.zeros([numk])
    kx = np.zeros([numk])
    ky = np.zeros([numk])
    ksize = np.zeros([numk])
    CONNECTIVITY = 4  # values can be 4 or 8
    M = compactness  # 10.0;
    invwt = (M * M * numk) / sz
    qlength = que.qsize()
    pixelcount = 0

    # Run main loop
    while qlength > 0:
        node = que.get()
        qlength -= 1
        k = node.k
        x = node.i >> 16 & 0xffff
        y = node.i & 0xffff
        i = y * width + x
        if labels[i] < 0:
            labels[i] = k
            pixelcount += 1
            kl[k] += lv[i]
            ka[k] += av[i]
            kb[k] += bv[i]
            kx[k] += x
            ky[k] += y
            ksize[k] += 1.0
            for p in range(CONNECTIVITY):
                xx = x + dx8[p]
                yy = y + dy8[p]
                if not (xx < 0 or xx >= w or yy < 0 or yy >= h):
                    ii = i + dn8[p]
                    if labels[ii] < 0:  # create new nodes
                        ldiff = kl[k] - lv[ii] * ksize[k]
                        adiff = ka[k] - av[ii] * ksize[k]
                        bdiff = kb[k] - bv[ii] * ksize[k]
                        xdiff = kx[k] - xx * ksize[k]
                        ydiff = ky[k] - yy * ksize[k]

                        colordist = ldiff * ldiff + adiff * adiff + bdiff * bdiff
                        xydist = xdiff * xdiff + ydiff * ydiff
                        slicdist = (colordist + xydist * invwt) / (
                        ksize[k] * ksize[k])  # late normalization by ksize[k], to have only one division operation

                        temp_i = int(int(xx) << 16 | int(yy))
                        temp_k = k
                        temp_d = slicdist
                        tempnode = NODE(temp_i, temp_k, temp_d)
                        que.put(tempnode)
                        qlength += 1

    print('Run main loop: %f' % (time.time() - start))
    start = time.time()

    outnumk = numk
    # Label the rarely occuring unlabelled pixels
    if labels[0] < 0:
        labels[0] = 0

    for y in range(1, height):
        for x in range(1, width):
            i = y * width + x
            if labels[i] < 0:
                if labels[i - 1] >= 0:
                    labels[i] = labels[i - 1]
                elif labels[i - width] >= 0:
                    labels[i] = labels[i - width]
    print('Label the rarely occuring unlabelled pixels: %f' % (time.time() - start))
    return labels, outnumk


if __name__ == "__main__":
    start = time.time()
    im = cv2.imread("bee1.jpg")
    lab = rgb2lab(im, 3)
    lvec = lab[:, :, 0]
    avec = lab[:, :, 1]
    bvec = lab[:, :, 2]
    h, w, c = im.shape
    lvec = lvec.reshape([h * w])
    avec = avec.reshape([h * w])
    bvec = bvec.reshape([h * w])
    print('rgb2lab:%f' % (time.time() - start))
    start = time.time()

    innumk = 200
    compactness = 20.0
    labels, outnumk = SNIC(lvec, avec, bvec, w, h, innumk, compactness)
    labels = labels.reshape([h, w])
    label_im = color.label2rgb(labels)
    cv2.imshow('show', label_im)
    print('SNIC:%f' % (time.time() - start))
    cv2.waitKey(0)
