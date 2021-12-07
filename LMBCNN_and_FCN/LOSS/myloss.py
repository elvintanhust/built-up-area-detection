from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import losses
import numpy as np
import tensorflow as tf
# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 10e-8

def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  x = ops.convert_to_tensor(x)
  if x.dtype != dtype:
    x = math_ops.cast(x, dtype)
  return x

def commission(y_true, y_pred):
    pos = math_ops.cast((y_pred >= 0.5), tf.float32)
    neg = math_ops.cast((y_pred < 0.5), tf.float32)
    PT = K.sum(pos * y_true)
    PF = K.sum(pos * (1 - y_true))
    com = PF / (PT + PF + 0.0)
    return com

def omission(y_true, y_pred):
    pos = math_ops.cast((y_pred >= 0.5), tf.float32)
    neg = math_ops.cast((y_pred < 0.5), tf.float32)
    PT = K.sum(pos * y_true)
    NF = K.sum(neg * y_true)
    omi = NF / (PT + NF + 0.0)
    return omi

def accuracy(y_true, y_pred):
    pos = math_ops.cast((y_pred >= 0.5), tf.float32)
    neg = math_ops.cast((y_pred < 0.5), tf.float32)
    PT = K.sum(pos * y_true)
    PF = K.sum(pos * (1 - y_true))
    NT = K.sum(neg * (1-y_true))
    NF = K.sum(neg * y_true)
    acc = (PT+NT) / (PT + NT+PF+NF + 0.0)
    return acc

def FCN_loss(y_true, y_pred):
    y_pred_bin = math_ops.cast((y_pred>=0.5), tf.float32)
    commission = y_pred_bin*(1-y_true)
    omission = (1-y_pred_bin)*y_true
    rightpix = y_pred_bin*y_true + (1-y_pred_bin)*(1-y_true)
    SE = K.square(y_pred - y_true)
    lossTemp = rightpix*SE + commission*SE*1 + omission*SE*1
    return K.mean(lossTemp, axis=-1)

def NCA_loss(y_true, y_pred):
    y_true_T = tf.transpose(y_true)
    y_pred_T = tf.transpose(y_pred)
    gt = tf.matmul(y_true,y_true_T)
    distMat = tf.matmul(y_pred,y_pred_T)
    return tf.reduce_sum(tf.multiply(gt,distMat))/tf.reduce_sum(distMat)


def focal_loss(y_true,y_pred):
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= math_ops.reduce_sum(
        y_pred, axis=len(y_pred.get_shape()) - 1, keep_dims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    return -math_ops.reduce_sum(
        0.25 * math_ops.square(1 - y_true * y_pred)*y_true * math_ops.log(y_pred),
        axis=len(y_pred.get_shape()) - 1)


def metric_loss(y_true, y_pred):
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= math_ops.reduce_sum(
        y_pred, axis=len(y_pred.get_shape()) - 1, keep_dims=True)
    # manual computation of crossentropy
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon, 1. - epsilon)

    return -math_ops.reduce_sum(
        0.25 * math_ops.square(1 - y_true * y_pred) * y_true * math_ops.log(y_pred),
        axis=len(y_pred.get_shape()) - 1)


"""
focal_loss已经添加到keras的定义文件中，具体如下：

    r'D:\MyProject\Tensorflow\Anaconda3\envs\tensorflow\Lib\site-packages\tensorflow\contrib\keras\python\keras\losses.py'中添加如下代码：
        def focal_loss(y_true, y_pred):
            return K.focal_loss(y_pred, y_true)
    
    r'D:\MyProject\Tensorflow\Anaconda3\envs\tensorflow\Lib\site-packages\tensorflow\contrib\keras\python\keras\backend.py'中添加如下代码：
        def focal_loss(y_true, y_pred):
          # scale preds so that the class probas of each sample sum to 1
          y_pred /= math_ops.reduce_sum(
            y_pred, axis=len(y_pred.get_shape()) - 1, keep_dims=True)
          # manual computation of crossentropy
          epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
          y_pred = clip_ops.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
          return -math_ops.reduce_sum(
            0.25 * math_ops.square(1 - y_true * y_pred) * y_true * math_ops.log(y_pred),
            axis=len(y_pred.get_shape()) - 1)

"""