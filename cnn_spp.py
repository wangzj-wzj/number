# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
#图像读取库
from PIL import Image
from skimage import transform,io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#矩阵运算库

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
#train = False


def spp_layer(input_, levels=3, name = 'SPP_layer',pool_type = 'max_pool'):

    '''
    Multiple Level SPP layer.
    
    Works for levels=[1, 2, 3, 6].
    '''
    
    shape = input_.get_shape().as_list()
    #print(shape)
    
    with tf.variable_scope(name):

        for l in range(levels):
        #设置池化参数
            l = l + 1
            ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
            
            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                #print(pool)
                pool = tf.reshape(pool,(shape[0],-1),)
                
            else:
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1:
                x_flatten = tf.reshape(pool,(shape[0],-1))
            else:
                x_flatten = tf.concat((x_flatten,pool),axis=1) #四种尺度进行拼接
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
            

    return x_flatten



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

#X_dropout_rate = 0
pool2_dropout_rate =0

n_fc1 = 600
fc1_dropout_rate = 0.5
n_outputs = 10
#learning_rate = 0.001
batch_size = 500

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs, channels], name="X")
    X_reshaped = tf.reshape(X, shape=[batch_size, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')


with tf.name_scope('cnn'):
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,strides=conv1_stride, padding=conv1_pad, activation=None, name="conv1")
    conv1_bn = tf.layers.batch_normalization(conv1,training=training)
    conv1_bn_act = tf.nn.relu(conv1_bn)
#    conv1_bn_act = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,strides=conv2_stride, padding=conv2_pad,activation=None, name="conv2")
    conv2_bn = tf.layers.batch_normalization(conv2,training=training)
    conv2_bn_act = tf.nn.relu(conv2_bn)
   #print(conv1_bn_act.shape,pool1.shape,conv2_bn_act.shape)
#    pool2 = tf.nn.max_pool(conv2_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
#    pool2_flat = tf.layers.flatten(pool2)
#    pool2_flat_drop = tf.layers.dropout(pool2_flat, pool2_dropout_rate, training=training)
#    print(pool2_flat_drop)
    spp = spp_layer(conv2_bn_act,4)
    print(spp.shape)

    fc1 = tf.layers.dense(spp, n_fc1, activation=None, name="fc1")
    fc1_bn =  tf.layers.batch_normalization(fc1,training=training)
    fc1_bn_act = tf.nn.relu(fc1_bn)
    fc1_drop = tf.layers.dropout(fc1_bn_act, fc1_dropout_rate, training=training)
    #print(fc1.shape,fc1_bn_act.shape)

with tf.name_scope("output"):
    logits = tf.layers.dense(spp, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    #print(Y_proba.shape)

with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, n_outputs))
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): #要先执行完update_ops操作后才能开始学习
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
#        print(image.shape)
        image = transform.resize(image, (height, width))
        label = int(fname.split("_")[0])
        datas.append(image)
        labels.append(label)
#        print('reading the images:%s' % fpath)

    datas = np.array(datas)
    labels = np.array(labels)

#    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

fpaths, X_train, y_train  = read_data(data_dir)
fpaths, X_test, y_test = read_data(test_dir)

X_train = X_train.astype(np.float32).reshape(-1, height*width, 3) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, height*width, 3) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:500], X_train[500:]
y_valid, y_train = y_train[:500], y_train[500:]
X_test = X_test[:500]
y_test = y_test[:500]


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
#batch_size = 200
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
#                sess.run(clip_all_weights)
                if iteration % check_interval == 0:
                    loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                    if loss_val < best_loss_val:
                        best_loss_val = loss_val
                        checks_since_last_progress = 0
                        best_model_params = get_model_params()
                    else:
                        checks_since_last_progress += 1
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_batch * 100, acc_val * 100, best_loss_val))
            if checks_since_last_progress > max_checks_without_progress:
                print("Early stopping!")
                break

        if best_model_params:
            restore_model_params(best_model_params)
#        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Final train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, Final accuracy on test set: {:.4f}%".format(acc_batch * 100, acc_val * 100, acc_test * 100))
        save_path = saver.save(sess, model_path)
    else:
        print("测试模式")
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9"
        }
        predicted_labels_val = sess.run(Y_proba, feed_dict={X: X_test, y: y_test})
        for fpath, real_label, predicted_label in zip(fpaths, y_test, predicted_labels_val):
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[np.argmax(predicted_label)]
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))

#        for X_batch, y_batch in shuffle_batch(X_test, y_test, 1):
#            st = time.time()
#            predicted_labels_val=sess.run(Y_proba, feed_dict={X: X_batch, y: y_batch})
#            print('Elapsed time: {}'.format(time.time()-st))
#            predicted_label_name = label_name_dict[np.argmax(predicted_labels_val)]
#            print("{} => {}".format(y_batch, predicted_label_name))


