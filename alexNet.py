# AlexNet实现
import tensorflow as tf
import numpy as np
import os
import cv2
import caffe_classes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略烦人的警告


# 卷积层、池化层、连接层的设置
# group=2时等于AlexNet分上下两部分
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)


def dropout(x, keepPro, name=None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)


def LRN(x, R, alpha, beta, name=None, bias=1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.get_variable("b", [outputD], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding="SAME", groups=1):  # group为2时等于AlexNet中分上下两部分
    """convlutional"""
    channel = int(x.get_shape()[-1])  # 获取channel
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)  # 定义卷积的匿名函数
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])
        b = tf.get_variable("b", shape=[featureNum])

        xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)  # 划分后的输入和权重
        wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]  # 分别提取feature map
        mergeFeatureMap = tf.concat(axis=3, values=featureMap)  # feature map整合
        # print mergeFeatureMap.shape
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)  # relu后的结果


# alexNet模型
class alexNet(object):
    """alexNet model"""

    def __init__(self, x, keepPro, classNum, skip, modelPath="bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        # build CNN
        self.buildCNN()

    def buildCNN(self):  # 创建网络
        """build model"""
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", groups=2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    def loadModel(self, sess):  # 加载模型
        """load model"""
        wDict = np.load(self.MODELPATH, encoding="bytes").item()
        # for layers in model
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse=True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            # bias
                            sess.run(tf.get_variable('b', trainable=False).assign(p))
                        else:
                            # weights
                            sess.run(tf.get_variable('w', trainable=False).assign(p))

# 测试
def res(path):
    # some params
    dropoutPro = 1
    classNum = 1000
    skip = []
    # get testImage
    testImg = cv2.imread(path)

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.loadModel(sess)  # 加载模型

        # img preprocess
        test = cv2.resize(testImg.astype(np.float), (227, 227))  # resize成网络输入大小
        test -= imgMean  # 去均值
        test = test.reshape((1, 227, 227, 3))  # 拉成tensor
        maxx = np.argmax(sess.run(softmax, feed_dict={x: test}))
        res = caffe_classes.class_names[maxx]  # 取概率最大类的下标
        print(res)


if __name__ == '__main__':
    path = "image_test/0.jpg"
    res(path)
    pass