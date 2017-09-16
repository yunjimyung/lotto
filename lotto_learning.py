import tensorflow as tf
import numpy as np
import datetime

time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tf.set_random_seed(777)

xy = np.loadtxt('./lotto_trainingset.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./lotto_testset.csv', delimiter=',', dtype=np.float32)

#Evaluation model
x_testset = test_data[:,0:-2]
y_testset = test_data[:,-2:]
x_dataset = xy[:,0:-2]
y_dataset = xy[:,-2:]

#print(x_dataset.shape, x_dataset, len(x_dataset))
#print(y_dataset.shape, y_dataset)

#set pramater
learning_rate = 0.001
training_epoch = 20
batch_size = 100
data_size = 539
keep_prob_number = 0.7

# input place holders
X = tf.placeholder(tf.float32, [None, 49])
X_img = tf.reshape(X, [-1, 7, 7, 1])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

#convolution neural network
# L1 ImgIn shape=(?, 7, 7, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01))
#    Conv     -> (?, 7, 7, 32)
#    Pool     -> (?, 4, 4, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 4, 4, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))
#    Conv      ->(?, 4, 4, 64)
#    Pool      ->(?, 4, 4, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2_flat = tf.reshape(L2, [-1, 2 * 2 * 32])

# Final FC 2x2x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[2 * 2 * 32, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L2_flat, W3) + b
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

count_epoch=[]
cost_value = []
# train model
print('Learning started. It takes sometime.')
for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(data_size/batch_size)
    i = 0
    for j in range(int(data_size/batch_size+1)):
        k = i
        i += batch_size
        x_data = xy[k:i, 0:-2]
        y_data = xy[k:i, -2:]
        c,_ = sess.run([cost, optimizer], feed_dict={X:x_data, Y:y_data, keep_prob:keep_prob_number})
        avg_cost += c / total_batch

#print result 2
#   if epoch % 10 == 0:
    print('Epoch:', '%04d'%(epoch + 1), 'Cost:', '{:.9f}'.format(avg_cost))
    cost_value.append(avg_cost)
    count_epoch.append(epoch)
#        with open('./save/result.txt','a') as f:

#           f.write("\nEpoch: %d  Cost: %f" %(epoch + 1, avg_cost))
print("Learning finished")

# Test model and check accuracy
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print('Accuracy:', sess.run(accuracy, feed_dict={
#      X: x_testset, Y: y_testset,keep_prob: 1}))
print('Accuracy:', sess.run(logits, feed_dict={
      X: x_testset, Y: y_testset,keep_prob: 1}))

#Save model
saver = tf.train.Saver()
save_path = saver.save(sess, './save/training')
#print('Model saved in file:', save_path)

'''
#print result
x = prediction_testset.astype(np.int64)
x = x.reshape(1,67)
y = y_testset.reshape(1,67)
with open('./save/result.txt','a') as f:
   f.write('\n\ntestdata = ')
   f.write('\n%s'%str(y))
   f.write('\n\nprediction = ')
   f.write('\n%s'%str(x))
   f.write('\n\ntestdata - prediction = ')
   f.write('\n%s'%str(y-x))
'''
import matplotlib.pyplot as plt
plt.plot(count_epoch, cost_value, 'g-')
plt.show()
