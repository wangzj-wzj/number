import tensorflow as tf

initial = tf.truncated_normal([5,5,1,32], stddev=0.1)



sess = tf.Session()
ov = sess.run(initial)
print(ov)
