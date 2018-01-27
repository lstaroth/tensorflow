from datetime import datetime
import math
import time
import tensorflow as tf

#设置参数
batch_size=32
num_batches=100


def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())

def inference(images):
    parameters=[]

    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights')
        conv1=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv1,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters+=[kernel,biases]


    pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1])
