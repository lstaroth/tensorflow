from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32, [None, 784])
y_=tf.placeholder(tf.float32, [None,10])

#生成权重
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),dtype=tf.float32)

#生成偏置
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape),dtype=tf.float32)

#卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积层
W_conv1=weight_variable([5, 5, 1, 32])
b_conv1=bias_variable([32])
x_image=tf.reshape(x, [-1,28,28,1])
h_conv1=tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#第二层卷积层
W_conv2=weight_variable([5, 5, 32, 64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#第三层全链接层
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2, [-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#第四层dropout层
drop_rate=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1, drop_rate)

#第五层全链接层
W_fc2=weight_variable([1024, 10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#损失函数
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))

#训练方式
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#正确率计算
correct_prediction=tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#参数对象初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
num_train=20
save_path=str(num_train)+"/save_net.ckpt"

#循环训练
for i in range(num_train):
    batch = mnist.train.next_batch(50)
    if i%1 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], drop_rate: 1.0})
      print("step %d, training accuracy %f"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], drop_rate: 0.5})

#测试集上计算正确率
accuracyResult = list(range(10))
for i in range(10):
    batch = mnist.test.next_batch(1000)
    accuracyResult[i] = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],drop_rate:1.0})
print("Test accuracy:", numpy.mean(accuracyResult))

issaving=input("Do you want to save the model?(y/n)")
if issaving=='y':
    saver.save(sess,save_path)
    print("model has saved at "+save_path)
