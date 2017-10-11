import csv
import random
import numpy as np
import os
import tensorflow as tf
import datetime

def make_data():
    with open('./lotto_data.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            lst = []
            for i in range(51):
                lst.append(0)
            for j in row:
                lst[int(j)-1] = 1
                lst[49] = 1
            with open('./lotto_learning.csv','a',newline ='') as w:
                Writer = csv.writer(w)
                Writer.writerow(lst)

#matke prediction lotto number1
def make_number1():
    number_list = []
    while len(number_list) < 6:
         random_number = random.randrange(1,46)
         if random_number in number_list:
             continue
         number_list.append(random_number)
         number_list.sort()
    result = number_list
    return result

def make_number2():
    with open('./lotto_data.csv','r') as f:
        reader = csv.reader(f)
        temp1 = make_number1()
        for row in reader:
            temp2 = [int(i)for i in row]
            temp3 = list(set(temp1) - set(temp2))
            while len(temp3) < 3:
                temp1 = make_number1()
                temp3 = list(set(temp1) - set(temp2))
        return temp1
'''
def make_result():
    b = []
    for i in range(100):
        a = make_number2()
        b.append(a)
        lst = []
        for i in range(51):
            lst.append(0)
        for j in a:
            lst[int(j)-1] = 1
            lst[49] = 1
        with open('./lotto_resultset.csv','a',newline ='') as w:
            Writer = csv.writer(w)
            Writer.writerow(lst)
    c = np.array(b)
#    print(c)
    print(c[0,:])
#    os.remove('./lotto_resultset.csv')
'''

if __name__ == "__main__":

#convoluti neural network
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    tf.set_random_seed(777)

    test_data = np.loadtxt('./lotto_testset.csv', delimiter=',', dtype=np.float32)

    #Evaluation model
    x_testset = test_data[:,0:-2]
    y_testset = test_data[:,-2:]

    #set pramater
    learning_rate = 0.001
    training_epoch = 20
    batch_size = 100
    data_size = 230
    keep_prob_number = 0.7

    # input place holders
    X = tf.placeholder(tf.float32, [None, 49])
    X_img = tf.reshape(X, [-1, 7, 7, 1])
    Y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    #convolution neural network
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 16], stddev=0.01))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


    W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    L2_flat = tf.reshape(L2, [-1, 2 * 2 * 32])

    # Final FC 2x2x64 inputs -> 2 outputs
    W3 = tf.get_variable("W3", shape=[2 * 2 * 32, 2],
                         initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([2]))
    logits = tf.matmul(L2_flat, W3) + b

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = saver.restore(sess, './save/training')

    test_error = sess.run(logits, feed_dict={X: x_testset, Y: y_testset,keep_prob: 1})
#    print(test_error)
#    print(np.mean(test_error,0))

#prediction
    b = []
    for i in range(500):
        a = make_number2()
        b.append(a)
        lst = []
        for i in range(51):
            lst.append(0)
        for j in a:
            lst[int(j)-1] = 1
            lst[49] = 1
        with open('./lotto_resultdata.csv','a',newline ='') as w:
            Writer = csv.writer(w)
            Writer.writerow(lst)
    c = np.array(b)


#    print(c)
#    print(c[0,:])
#    os.remove('./lotto_resultset.csv')

    result_data= np.loadtxt('./lotto_resultdata.csv', delimiter=',', dtype=np.float32)
    x_result_data = result_data[:,0:-2]
    y_result_data = result_data[:,-2:]

    result_error = sess.run(logits, feed_dict={X: x_result_data, Y: y_result_data,keep_prob: 1})
    rmse_error = np.sqrt(np.square(np.mean(test_error,0)-result_error))

    number = np.argmin(rmse_error,0)[0]
    result_number = c[number,:]
#    print(np.argmin(rmse_error,0))
    print(number)
    print(result_number)
    with open('./save/result.txt','a') as f:
       f.write('\n%s  %s'%(time_now, str(result_number)))
    os.remove('./lotto_resultdata.csv')
