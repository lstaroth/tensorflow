import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#返回合适的初始化权重
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in+fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

#自编码器类
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input         #784
        self.n_hidden=n_hidden       #200
        self.transfer=transfer_function   #softplus
        self.scale=tf.placeholder(tf.float32)    #0.1
        self.training_scale=scale
        self.weights=self._initialize_weights()
        self.x=tf.placeholder(tf.float32,[None,self.n_input])    #输入数据占位符
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))   #输入加上噪声在转换到隐藏层
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])     #重建层
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))             #损失函数等于重建层和原始数据的差的平方和
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

#初始化全局权重，返回权重字典集
    def _initialize_weights(self):
        all_weights=dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

#返回损失并训练
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

#只返回损失
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

#获取隐藏层的值
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})    #用输入的X作为feed数据，返回运算完成后的hidden层的值

#得到隐藏层的输入通过重建层还原
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])


    def getBiases(self):
        return self.sess.run(self.weights['b1'])

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#先对训练数据进行归一化处理
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)  #训练数据归一化
    X_train=preprocessor.transform(X_train)          #将归一化后的训练数据转换为隐藏层替换掉X_train
    X_test=preprocessor.transform(X_test)            #将测试数据转换为隐藏层数据
    return X_train,X_test


#接收训练集和分支需要的数量，返回一个随机的指定数量的子集
def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)


#实例化类对象autoencoder
autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,     #输出变量784个
                                             n_hidden=200,    #抽象出200个高阶特征
                                             transfer_function=tf.nn.softplus,
                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                             scale=0.01)

n_samples=int(mnist.train.num_examples)
training_epochs=100          #最大训练步数
batch_size=128               #分支大小
display_step=1               #每隔几步显示一次


#开始训练
for epoch in range(training_epochs):
    avg_cost=0.
    total_batch=int(n_samples/batch_size)    #共多少分支
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)     #得到随机分支
        cost=autoencoder.partial_fit(batch_xs)           #拿到当前的损失数据并进行基于该分支的一次训练
        avg_cost+=cost/n_samples*batch_size

    if epoch % display_step ==0:          #判断是否需要显示
        print("Epoch:",'%04d' % (epoch + 1),"cost=","{:.9f}".format(avg_cost))

print("Total cost: "+str(autoencoder.calc_total_cost(X_test)))