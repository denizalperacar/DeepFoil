import math
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
import time


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# import inputs
#with open('polars.pickle', 'r') as fid:
#    polars = pickle.load(fid)
with open('af_points.pickle', 'rb') as fid:
    af_data_dic = pickle.load(fid, encoding='latin1')
with open('af_labels.pickle', 'rb') as fid:
    af_label = pickle.load(fid, encoding='latin1')
with open('label_afs.pickle', 'rb') as fid:
    label_af = pickle.load(fid, encoding='latin1')

# hyperparameters
lamda = 0.00001
learning_rate = 0.0001
epochs = 5000
batch_size = 1024
beta_1 = 0.95
beta_2 = 0.99
layers = [512, 256, 200, 128, 4]
s_train, s_dev, s_tes = 0.9, 0.05, 0.05

alpha_range= [-5, 15]
Re_range=[500000,10000000]
inputs_list=['af', 're', 'a']
outputs_list=['cl', 'cd', 'cdp', 'cm']
normalize= ['re', 'a', 'cl', 'cd', 'cdp', 'cm']
n_params = {}

# create input
data = pd.read_csv('raw_af_data.txt')
index = data.index
h = data.copy()
h = h[(h['a'] > alpha_range[0]) & (h['a'] < alpha_range[1])]
h = h[(h['re'] > Re_range[0]) & (h['re'] < Re_range[1])]
# h = h[(h['af'] != 382) & (h['af'] != 86) & (h['af'] != 19)]

# print(h.head(30))
# normalize re and alpha
# ====================================
for col in normalize:
    mu = h[col].mean()
    sigma = h[col].std()
    h[col] = (h[col] - mu) / sigma
    n_params[col] = {'mu':mu, 'sigma':sigma}
# ====================================
# shuffle the data three times
h = h.sample(frac=1, axis=0).reset_index(drop=True)
h = h.reindex(np.random.permutation(h.index)).reset_index(drop=True)
# third shuffle
inputs_train = h.sample(frac=s_train)
remaining = h.drop(inputs_train.index)
inputs_train = inputs_train.reset_index(drop=True)
inputs_dev = remaining.sample(frac=(s_dev/(1-s_train)))
inputs_test = remaining.drop(inputs_dev.index).reset_index(drop=True)
inputs_dev = inputs_dev.reset_index(drop=True)


x_train = inputs_train[['af', 're', 'a']].values.transpose()
y_train = inputs_train[['cl', 'cd', 'cdp', 'cm']].values.transpose()
#y_train = inputs_train[['cd']].values.transpose()
m_train = x_train.shape[1]

x_dev = inputs_dev[['af', 're', 'a']].values.transpose()
#y_dev = inputs_dev[['cd']].values.transpose()
y_dev = inputs_dev[['cl', 'cd', 'cdp', 'cm']].values.transpose()
m_dev = x_dev.shape[1]

x_test = inputs_test[['af', 're', 'a']].values.transpose()
#y_test = inputs_test[['cd']].values.transpose()
y_test = inputs_test[['cl', 'cd', 'cdp', 'cm']].values.transpose()
m_test = x_test.shape[1]

# NETWORK DEFINITION

n = af_data_dic[label_af[0]]['input'].shape[1] + 2
x = tf.placeholder('float64', shape=(n, None))
y = tf.placeholder('float64', shape=(y_train.shape[0], None))

#tf.reset_default_graph()
W1 = tf.get_variable("W1", (layers[0], n),
                        initializer=tf.contrib.layers.xavier_initializer(seed=0),
                        dtype=tf.float64)

b1 = tf.get_variable("b1", [layers[0], 1], initializer=tf.zeros_initializer(), dtype=tf.float64)

W2 = tf.get_variable("W2", [layers[1], layers[0]],
                        initializer=tf.contrib.layers.xavier_initializer(seed=0),
                        dtype=tf.float64)

b2 = tf.get_variable("b2", [layers[1], 1], initializer=tf.zeros_initializer(), dtype=tf.float64)

W3 = tf.get_variable("W3", [layers[2], layers[1]],
                        initializer=tf.contrib.layers.xavier_initializer(seed=0),
                        dtype=tf.float64)

b3 = tf.get_variable("b3", [layers[2], 1], initializer=tf.zeros_initializer(), dtype=tf.float64)

W4 = tf.get_variable("W4", [layers[3], layers[2]],
                        initializer=tf.contrib.layers.xavier_initializer(seed=0),
                        dtype=tf.float64)

b4 = tf.get_variable("b4", [layers[3], 1], initializer=tf.zeros_initializer(), dtype=tf.float64)

W5 = tf.get_variable("W5", [layers[4], layers[3]],
                        initializer=tf.contrib.layers.xavier_initializer(seed=0),
                        dtype=tf.float64)

b5 = tf.get_variable("b5", [layers[4], 1], initializer=tf.zeros_initializer(), dtype=tf.float64)

# Forward Prop
Z1 = tf.add(tf.matmul(W1, x), b1)
A1 = tf.nn.leaky_relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.leaky_relu(Z2)
Z3 = tf.add(tf.matmul(W3, A2), b3)
A3 = tf.nn.leaky_relu(Z3)
Z4 = tf.add(tf.matmul(W4, A3), b4)
A4 = tf.nn.leaky_relu(Z4)
Z5 = tf.add(tf.matmul(W5, A4), b5)
A5 = Z5

# compute cost
cost = tf.reduce_sum((y-A5)**2.)

# frobenious regularization
reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)

# compute reqularized loss
cost = tf.reduce_mean(cost + lamda * reg)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2).minimize(cost)
init = tf.global_variables_initializer()

# training
saver = tf.train.Saver(max_to_keep=4)
seed = 10
costs = []

fid = open('res.txt', 'a')
fid.write("\n\n\n training {} \n\n".format(time.time()))
fid.close()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

    for epoch in range(epochs):
        epoch_cost = 0.
        n_batch = int(m_train/batch_size)
        seed = seed + 1
        minibatches = random_mini_batches(x_train, y_train, batch_size, seed)
        for minibatch in minibatches:
            (minibatch_x, minibatch_y) = minibatch
            # convert the x_s
            x_af = minibatch_x[0, :].astype(int)
            x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
            x_temp = minibatch_x[1:3, :]
            minibatch_x = np.concatenate((x_af, x_temp), axis=0)
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={x: minibatch_x, y: minibatch_y})
            epoch_cost += minibatch_cost / n_batch

        # Print the cost every epoch
        if epoch % 1 == 0:
            fid = open('res.txt', 'a')
            fid.write("Cost after epoch {}: {}\n".format(epoch, epoch_cost))
            fid.close()
        if epoch % 1 == 0:
            costs.append(epoch_cost)


        if epoch % 10 == 0:
            cp = saver.save(sess, 'cp/cp.ckpt')

            # create x_dev
            x_af = x_dev[0, :].astype(int)
            x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
            x_temp = x_dev[1:3, :]
            x_dex_n = np.concatenate((x_af, x_temp), axis=0)

            # predict the error
            # predict = tf.sqrt(tf.reduce_mean(tf.squared_difference(A4, y)))
            # minimize the infinity norm
            # predict = tf.math.reduce_max(tf.sqrt(tf.squared_difference(A4, y)))
            predict = tf.abs(y - A5)
            error = sess.run(predict, feed_dict={x: x_dex_n, y: y_dev})
            # print(error.shape)
            # time.sleep(1)
            fid = open('res.txt', 'a')
            fid.write('infinity norm:\n\n')
            fid.write('infinity norm cl : {}\n'.format(error[0, :].max()))
            fid.write('infinity norm cd : {}\n'.format(error[1, :].max()))
            fid.write('infinity norm cdp: {}\n'.format(error[2, :].max()))
            fid.write('infinity norm cm : {}\n'.format(error[3, :].max()))
            fid.write('cl  std {} mean {}\n'.format(error[0, :].std(), error[0, :].mean()))
            fid.write('cd  std {} mean {}\n'.format(error[1, :].std(), error[1, :].mean()))
            fid.write('cdp std {} mean {}\n'.format(error[2, :].std(), error[2, :].mean()))
            fid.write('cm  std {} mean {}\n\n'.format(error[3, :].std(), error[3, :].mean()))
            fid.close()

            # plot parameters
            # trouble = x_dev[:, error.argmax()]
            # points = []
            # for i in range(1000):
            #    points.append(sess.run(A5, feed_dict={x: x_dex_n[:, i].reshape(-1, 1)}))
            # hh = np.abs(y_dev[:, :1000].reshape(-1, 1).transpose() - np.array(points).flatten())
            # points = []
            # plt.plot(range(1000), np.abs(hh).transpose(), 'r')
            # #plt.plot(range(x_dex_n.shape[1]), y_dev.transpose(), 'b')
            # plt.show()
            # fid = open('res.txt', 'a')
            # fid.write('infinity norm params: \nairfoil {} {} alpha {}  Re {}\n'.format(int(trouble[0]),
            #                                                                     label_af[int(trouble[0])],
            #                                                                     trouble[2] * sigma_a + mu_a,
            #                                                                     trouble[1] * re_max))
            # fid.write('std {} mean {}\n\n'.format(error.std(), error.mean()))
            # fid.close()
            # print('dev_set_error: ', error)

        if epoch % 20 == 0:
            x_af = x_test[0, :].astype(int)
            x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
            x_temp = x_test[1:3, :]
            x_test_n = np.concatenate((x_af, x_temp), axis=0)

            predict = tf.abs(y - A5)
            error = sess.run(predict, feed_dict={x: x_test_n, y: y_test})
            fid = open('res.txt', 'a')
            fid.write('infinity norm test cl : {}\n'.format(error[0, :].max()))
            fid.write('infinity norm test cd : {}\n'.format(error[1, :].max()))
            fid.write('infinity norm test cdp: {}\n'.format(error[2, :].max()))
            fid.write('infinity norm test cm : {}\n\n'.format(error[3, :].max()))
            fid.close()


    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # create x_dev
    x_af = x_dev[0, :].astype(int)
    x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
    x_temp = x_dev[1:3, :]
    x_dex_n = np.concatenate((x_af, x_temp), axis=0)

    # predict the error
    predict = tf.sqrt(tf.reduce_mean(tf.squared_difference(A5, y)))
    error = sess.run(predict, feed_dict={x: x_dex_n, y: y_dev})

    print('dev_set_error: ', error)
    
