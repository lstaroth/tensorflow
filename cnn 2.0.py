import tensorflow as tf
import numpy
from PIL import Image

# 添加x作为占位符
x = tf.placeholder("float", [1, 784])


# 生成权重函数
def weight_variable(shape):
    # tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正态分布，均值和标准差自己设定
    # 权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)


# 生成偏置函数
# 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)


# 卷积函数
# 卷积使用1步长，0边距的模板，池化用2x2的模板
def conv2d(x, W):
    # x：待卷积的矩阵具有[batch, in_height, in_width, in_channels]这样的shape
    # w：卷积核具有[filter_height, filter_width, in_channels, out_channels]这样的shape
    # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化函数
# 和卷积基本相同
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 卷积在每个5x5的patch中算出32个特征。
# 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，
# 接着是输出几个单位，和输出的几个维度
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# shape:[batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积+偏置，然后给relu激活函数，最后激活函数返回值池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32

# 第二层卷积，池化
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

# 全连接层1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 全连接层3
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
prediction = tf.argmax(y_conv, 1)

# 初始化数据读取模型
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "2000/save_net.ckpt")

imgmatrix = []
for i in range(10):
    img = Image.open('.//image//' + str(i) + '.tif')
    # imgmatrix.append(numpy.clip(numpy.array(img),0,1).reshape(1,784))
    imgmatrix.append(numpy.array(img).reshape(1, 784))

for i in range(10):
    result = sess.run(prediction, feed_dict={x: imgmatrix[i]})
    print(i, "The reslut is", result)
