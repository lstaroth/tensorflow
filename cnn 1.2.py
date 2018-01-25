from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32, [None, 784])#28*28
y_=tf.placeholder(tf.float32, [None,10])
keep_prob=tf.placeholder(tf.float32)

def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,dtype=tf.float32)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial,dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W_conv1=weight_variable([5, 5, 1, 6])
b_conv1=bias_variable([6])
x_image=tf.reshape(x, [-1,28,28,1])
h_conv1=tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5, 5, 6, 16])
b_conv2=bias_variable([16])
h_conv2=tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2=max_pool_2x2(h_conv2)

W_fc1=weight_variable([7*7*16,120])
b_fc1=bias_variable([120])
h_pool2_flat=tf.reshape(h_pool2, [-1,7*7*16])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)

W_fc2=weight_variable([120, 100])
b_fc2=bias_variable([100])
h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3=weight_variable([100, 10])
b_fc3=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
tf.summary.scalar('cross_entropy',cross_entropy)
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)
sess = tf.InteractiveSession()


sess.run(tf.global_variables_initializer())
for i in range(5001):
    batch = mnist.train.next_batch(50)
    if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

accuracyResult = list(range(10))
for i in range(10):
    batch = mnist.test.next_batch(1000)
    accuracyResult[i] = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
print("Test accuracy:", numpy.mean(accuracyResult))