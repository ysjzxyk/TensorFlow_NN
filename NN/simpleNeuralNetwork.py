#!/usr/bin/env python
#coding: UTF-8
from TensorFlow.mnist import input_data
import tensorflow as tf
#加载mnist数据集
mnist = input_data.read_data_sets('data/', one_hot=True) #在多类场景下，one_hot=true表示，只有一个元素的值是1，其他元素的值是0， 一个长度为n的数组，只有一个元素是1.0，其他元素是0.0。
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
# print(trainlabel[9])
print('mnist loaded')


# NETWORK TOPOLOGIES (网络结构)
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_classes = 10

# INPUTS AND OUTPUTS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# NETWORK PARAMETERS
stddev = 0.1  #学习率
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("NETWORK READY")



#前向传播
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])

# 预测
pred = multilayer_perceptron(x, weights, biases)

# LOSS AND OPTIMIZER
#tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
    #第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
    #第二个参数labels：实际的标签，大小同上

#反向传播
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) #损失函数：tf.nn.softmax_cross_entropy_with_logits交叉熵函数
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

training_epochs = 20
batch_size = 100
display_step = 4

# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # ITERATION
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch

    # DISPLAY
    if (epoch+1) % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print("TEST ACCURACY: %.3f" % (test_acc))
print("OPTIMIZATION FINISHED")