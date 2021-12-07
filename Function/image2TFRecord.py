"""
    将TXT文件记录的GF2样本转换为TFRecord格式
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from random import shuffle
import cv2


# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_data_city_or_noncity(f="train", city=True, path="E:\\DATA\\GF2_TensorFlow"):
    if f == "train":
        if city:
            pic_dir = r'E:\DATA\GF2\panchromatic64\train\city'
        else:
            pic_dir = r'E:\DATA\GF2\panchromatic64\train\noncity'
    elif f == "val":
        if city:
            pic_dir = r'E:\DATA\GF2\panchromatic64\val\city'
        else:
            pic_dir = r'E:\DATA\GF2\panchromatic64\val\noncity'
    elif f == "test":
        if city:
            pic_dir = r'E:\DATA\GF2\panchromatic64\test\city'
        else:
            pic_dir = r'E:\DATA\GF2\panchromatic64\test\noncity'
    else:
        return
    flielist = os.listdir(pic_dir)
    length = flielist.__len__()
    shuffle(flielist)
    # 每个文件写入多少数据
    instances_per_shard = 10000
    # 总共写入多少个文件
    num_shards = int((length - 1) / 10000) + 1
    # 记录文件写到第几个
    cursor = 0

    for i in range(num_shards):
        if city:
            filename = os.path.join(path, f, "city\\GF2-%.5d-of-%.5d.tfrecord" % (i, num_shards))
        else:
            filename = os.path.join(path, f, "noncity\\GF2-%.5d-of-%.5d.tfrecord" % (i, num_shards))
        with tf.python_io.TFRecordWriter(filename) as  writer:
            for j in range(instances_per_shard):
                if cursor < length:
                    dir = os.path.join(pic_dir,flielist[cursor])
                    # 读图像并解码
                    image = cv2.imread(dir)
                    image = image[:,:,0]
                    cursor = cursor + 1
                    print("%s 已处理：%.2f%%\r" % (f, cursor * 100 / length))
                else:
                    continue
                # plt.imshow(image)
                # plt.title(labels)
                # plt.show()
                # time.sleep(1)
                # plt.close()
                img_raw = image.tobytes()  # 将图片转化为二进制格式
                labels = int(1-city)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'im': _bytes_feature(img_raw),
                    'label': _int64_feature(labels)
                }))
                writer.write(example.SerializeToString())


if __name__ == "__main__":
    # write_data( f="train", path="E:\\DATA\\GF2_TensorFlow")
    # write_data(f="val", path="E:\\DATA\\GF2_TensorFlow")
    # write_data(f="test", path="E:\\DATA\\GF2_TensorFlow")
    write_data_city_or_noncity(f="train", city=True, path="E:\\DATA\\GF2_TensorFlow")
    write_data_city_or_noncity(f="train", city=False, path="E:\\DATA\\GF2_TensorFlow")
    write_data_city_or_noncity(f="test", city=True, path="E:\\DATA\\GF2_TensorFlow")
    write_data_city_or_noncity(f="test", city=False, path="E:\\DATA\\GF2_TensorFlow")
    write_data_city_or_noncity(f="val", city=True, path="E:\\DATA\\GF2_TensorFlow")
    write_data_city_or_noncity(f="val", city=False, path="E:\\DATA\\GF2_TensorFlow")

