#coding=utf-8

import os
#图像读取库
from PIL import Image
from skimage import transform,io
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg 
#矩阵运算库
import numpy as np
import tensorflow as tf

######
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#####

# 数据文件夹
data_dir = "./traindata/"
test_dir = './testdata/'
# 模型文件路径
model_path = "model/image_model"
train = True


# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
height = 75
width = 50
channels = 3
n_inputs = height * width

conv1_fmaps = 20
conv1_ksize = 5
conv1_stride = 2
conv1_pad = "SAME"

conv2_fmaps = 40
conv2_ksize = 4
conv2_stride = 2
conv2_pad = "SAME"

X_dropout_rate = 0.25
conv1_dropout_rate = 0
conv2_dropout_rate = 0
pool1_dropout_rate = 0
pool2_dropout_rate = 0.5

n_fc1 = 600
fc1_dropout_rate = 0.5
n_outputs = 10
learning_rate = 0.001


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs, channels], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int64, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(X_reshaped, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([4, 4, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([4 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


with tf.name_scope("train"):
#    loss = -tf.reduce_sum(y*tf.log(y_conv))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(y, n_outputs), logits=y_conv))
    optimizer = tf.train.AdamOptimizer()
#    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, n_outputs))
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = mpimg.imread(fpath)[:, :, :channels]
        image = transform.resize(image, (height, width))
        label = int(fname.split("_")[0])
        datas.append(image)
        labels.append(label)
        print('reading the images:%s' % fpath)

    datas = np.array(datas)
    labels = np.array(labels)

#    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels

X_train, y_train  = read_data(data_dir)
X_test, y_test = read_data(test_dir)



X_train = X_train.astype(np.float32).reshape(-1, height*width, 3) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, height*width, 3) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:500], X_train[500:]
y_valid, y_train = y_train[:500], y_train[500:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
#    print(len(X),rnd_idx)
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

n_epochs = 5000
batch_size = 100
iteration = 0

best_loss_val = np.infty
check_interval = 100
checks_since_last_progress = 0
max_checks_without_progress = 100
best_model_params = None 


with tf.Session() as sess:
    if train:
        print("训练模式")
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                iteration += 1
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
                if iteration % check_interval == 0:
                    loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                    if loss_val < best_loss_val:
                        best_loss_val = loss_val
                        checks_since_last_progress = 0
                        best_model_params = get_model_params()
                    else:
                        checks_since_last_progress += 1
#            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_train, y: y_train})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print("Epoch {}, train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_batch * 100, acc_val * 100, best_loss_val))
            if checks_since_last_progress > max_checks_without_progress:
                print("Early stopping!")
                break

        if best_model_params:
            restore_model_params(best_model_params)
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Final accuracy on test set:", acc_test)
        save_path = saver.save(sess, model_path)
    else:
        print("测试模式")
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))












