import tensorflow as tf
#Numpy是一个科学计算的工具包，通过randomstate生成模拟数据集
from numpy.random import RandomState

batch_size = 8 #神经网络训练集batch大小为8
#定义神经网络的结构，输入为2个参数，隐藏层为3个参数，输出为1个参数
#声明w1、w2两个变量，通过设定seed参数随机种子，随机种子相同，则每次使用此代码都生成相同的随机数
#stddev为标准差，没有mean设定均值，则均值默认为0
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1)) #w1为输入到隐藏层的权重，2*3的矩阵
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1)) #w2为隐藏层打输出的权重，3*1的矩阵

#维度中使用None，则可以不规定矩阵的行数，方便存储不同batch的大小。
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#定义神经网络前向传播的过程
a = tf.matmul(x, w1) #a为隐藏层的输出,matmul为矩阵的相乘
y = tf.matmul(a, w2) #y为神经网络的输出

#定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
#cross_entropy定义了真实值与预测值之间的交叉熵，是一种损失函数
train_step = tf.train.AdamOptimizer(0.001).minimize((cross_entropy)) #反向传播算法

#通过随机数生成一个模拟数据集
rdm = RandomState(1) #rdm为伪随机数发生器，种子为1，只要种子相同，该发生器每次生成的随机数都是一样的
dataset_size = 128
X = rdm.rand(dataset_size, 2) #生成随机数，大小为128*2的矩阵
#Y属于样本的标签，所有x1+x2<1的都被认为是正样本，其余为负样本。
Y = [[int(x1+x2 <1)] for (x1, x2) in X]  #列表解析格式
#若x1+x2 <1为真，则int(x1+x2 <1)为1，若假，则输出为0

#创建会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer() #所有需要初始化的值
    sess.run(init_op) #初始化变量
    print(sess.run(w1))
    print(sess.run(w2))

    '''
    #在训练之前神经网络权重的值
    w1 = [[-0.81131822, 1.48459876, 0.06532937], [-2.44270396, 0.0992484, 0.59122431]]
    w2 = [[-0.81131822, 1.48459876, 0.06532937]]
    '''

    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次从数据集中选8个数据进行训练
        start = (i * batch_size) % dataset_size  # 训练集在数据集中的开始位置
        end = min(start + batch_size, dataset_size)  # 结束位置，若超过dataset_size，则设为dataset_size

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 在训练之后神经网络权重的值
    print(sess.run(w1))
    print(sess.run(w2))