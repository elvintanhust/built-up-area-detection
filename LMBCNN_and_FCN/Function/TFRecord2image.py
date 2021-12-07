from Function.function import show_batch
import tensorflow as tf
import os
import numpy as np
import cv2
import random

root_path = 'E:\\DATA\\GF2_TensorFlow'
pic_path = 'E:\\DATA\\GF2'


def distorted_color(image, color_ordering=0):
    """
    给定一张图像，随机调整图像色彩。因为调整亮度、对比度、饱和度和色相的
    顺序会影响最后的结果，所以可以定义多种不同的顺序。训练时可以随机选择
    一种，进一步降低无关因素对模型的影响。
    :param image: 待处理图像
    :param color_ordering: 色彩调整顺序
    :return: 处理后图像
    """
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 4:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, width, height):
    """
    给定一张解码后的图像和目标图像尺寸以及标注框，此函数可以对给出的图像
    进行预处理。输入图像是识别问题中的原始训练图像，输出是训练时模型输入
    数据，预测时一般不需要随机变换。
    :param image: 待处理图像
    :param width: 输出图像宽
    :param height: 输出图像高
    :return:distorted_image: 处理后图像
    """
    # 没有标注框则认为整图是关注区

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 随机左右翻转图像
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    noise = tf.truncated_normal(image.shape, mean=0.0, stddev=0.01, dtype=tf.float32 )
    image = tf.add(image,noise)
    # 随机截取图像，减小需要关注物体大小对识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox,
                                                                      min_object_covered=0.9,
                                                                      aspect_ratio_range=[0.9, 1.1])
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将截取的图像调整为目标图像大小，缩放算法随机选择  method = np.random.randint(4)
    distorted_image = tf.image.resize_images(distorted_image, [height, width])
    distorted_image.set_shape([height, width,1])
    # 使用一种随机顺序调整图像色彩
    # distorted_image = distorted_color(distorted_image, np.random.randint(5))
    return tf.clip_by_value(distorted_image, 0.0, 1.0)



def next_city_batch(model='val', num_epochs=None, batch_size=32, out_height=128, out_width=128, n_classes=2,
                    isProcess=False):
    if model == 'val':
        dir = os.path.join(root_path, 'val')
    elif model == 'test':
        dir = os.path.join(root_path, 'test')
    elif model == 'train':
        dir = os.path.join(root_path, 'train')

    # 获取city数据和标签
    city_filenames = os.listdir(os.path.join(dir, 'city'))
    city_files = []
    for i in city_filenames:
        if '.tfrecord' == os.path.splitext(i)[1]:
            city_files.append(os.path.join(dir, 'city', i))
    city_filename_queue = tf.train.string_input_producer(city_files, num_epochs=num_epochs, shuffle=True,capacity=200)
    # 解析TFRecord文件里的数据
    city_reader = tf.TFRecordReader()
    _, city_serialized_example = city_reader.read(city_filename_queue)
    city_features = tf.parse_single_example(city_serialized_example, features={
        'im': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    city_image, city_label = city_features['im'], city_features['label']

    # 解析出像素矩阵
    city_decoded_image = tf.decode_raw(city_image, tf.uint8)
    # decoded_image.set_shape([height, width, channel])
    city_decoded_image = tf.reshape(city_decoded_image, [64, 64,1])

    if isProcess:
        city_decoded_image = preprocess_for_train(city_decoded_image, out_width, out_height)
    else:
        city_decoded_image = tf.image.convert_image_dtype(city_decoded_image, dtype=tf.float32)
        city_decoded_image = tf.image.resize_images(city_decoded_image, [out_height, out_width])
        city_decoded_image.set_shape([out_height, out_width,1])
        city_decoded_image = tf.clip_by_value(city_decoded_image, 0.0, 1.0)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    # 将处理后的图像和标签通过tf.train.shuffle_batch整理成训练所需batch
    if num_epochs == None:
        city_image_batch, city_label_batch = tf.train.shuffle_batch([city_decoded_image, city_label],
                                                                    batch_size=batch_size, capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue,allow_smaller_final_batch= False)
    else:
        city_image_batch, city_label_batch = tf.train.batch([city_decoded_image, city_label],
                                                            batch_size=batch_size, capacity=capacity,allow_smaller_final_batch= False)

    city_label_batch = tf.one_hot(city_label_batch, depth=n_classes)
    city_label_batch = tf.cast(city_label_batch, dtype=tf.float32)
    city_label_batch = tf.reshape(city_label_batch, [batch_size, n_classes])

    city_image_batch = tf.clip_by_value(city_image_batch, 0.0, 1.0)

    return city_image_batch, city_label_batch


def next_noncity_batch(model='val', num_epochs=None, batch_size=32, out_height=128, out_width=128, n_classes=2,
                       isProcess=True):
    # filename_queue = tf.train.string_input_producer(["E:\\DATA\\GF2_TensorFlow\\train\\GF2-00000-of-00040.tfrecord"])

    if model == 'val':
        dir = os.path.join(root_path, 'val')
    elif model == 'test':
        dir = os.path.join(root_path, 'test')
    elif model == 'train':
        dir = os.path.join(root_path, 'train')

    # 获取noncity数据和标签
    noncity_filenames = os.listdir(os.path.join(dir, 'noncity'))
    noncity_files = []
    for i in noncity_filenames:
        if '.tfrecord' == os.path.splitext(i)[1]:
            noncity_files.append(os.path.join(dir, 'noncity', i))
    noncity_filename_queue = tf.train.string_input_producer(noncity_files, num_epochs=num_epochs, shuffle=True,capacity=200)
    # 解析TFRecord文件里的数据
    noncity_reader = tf.TFRecordReader()
    _, noncity_serialized_example = noncity_reader.read(noncity_filename_queue)
    noncity_features = tf.parse_single_example(noncity_serialized_example, features={
        'im': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    noncity_image, noncity_label = noncity_features['im'], noncity_features['label']

    # 解析出像素矩阵
    noncity_decoded_image = tf.decode_raw(noncity_image, tf.uint8)
    noncity_decoded_image = tf.reshape(noncity_decoded_image, [64, 64,1])

    if isProcess:
        noncity_decoded_image = preprocess_for_train(noncity_decoded_image, out_width, out_height)
    else:
        noncity_decoded_image = tf.image.convert_image_dtype(noncity_decoded_image, dtype=tf.float32)
        noncity_decoded_image = tf.image.resize_images(noncity_decoded_image, [out_height, out_width])
        noncity_decoded_image.set_shape([out_height, out_width,1])
        noncity_decoded_image = tf.clip_by_value(noncity_decoded_image, 0.0, 1.0)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    if num_epochs == None:
        noncity_image_batch, noncity_label_batch = tf.train.shuffle_batch([noncity_decoded_image, noncity_label],
                                                                          batch_size=batch_size,
                                                                          capacity=capacity,
                                                                          min_after_dequeue=min_after_dequeue,allow_smaller_final_batch= False)
    else:
        noncity_image_batch, noncity_label_batch = tf.train.batch([noncity_decoded_image, noncity_label],
                                                                  batch_size=batch_size,
                                                                  capacity=capacity,allow_smaller_final_batch= False)

    noncity_label_batch = tf.one_hot(noncity_label_batch, depth=n_classes)
    noncity_label_batch = tf.cast(noncity_label_batch, dtype=tf.float32)
    noncity_label_batch = tf.reshape(noncity_label_batch, [batch_size, n_classes])

    noncity_image_batch = tf.clip_by_value(noncity_image_batch, 0.0, 1.0)

    return noncity_image_batch, noncity_label_batch


if __name__ == "__main__":

    with tf.Session() as sess:

        image_batch, label_batch = next_city_batch(model='test', num_epochs=2, batch_size=100, out_height=64,
                                                      out_width=64, n_classes=2,
                                                      isProcess=True)
        init_op = tf.local_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while 1:
            images, labels = sess.run([image_batch, label_batch])
            print('hello')
            # show_batch('show', images, showlabel=False, col=10, row=10, channel=1, predicts=None, labels=None)
        coord.request_stop()
        coord.join()
