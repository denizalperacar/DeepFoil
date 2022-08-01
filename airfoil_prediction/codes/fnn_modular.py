import math
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
import time


def load_data(af_points='af_points.pickle',
              af_labels='af_labels.pickle',
              label_afs='label_afs.pickle'):

    with open(af_points, 'rb') as fid:
        af_data_dic = pickle.load(fid, encoding='latin1')
    with open(af_labels, 'rb') as fid:
        af_label = pickle.load(fid, encoding='latin1')
    with open(label_afs, 'rb') as fid:
        label_af = pickle.load(fid, encoding='latin1')

    return af_data_dic, af_label, label_af


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


def hyper_parameters(lamda=0.00001, learning_rate=0.0001,
                     epochs=5000, batch_size=1024,
                     beta_1=0.95, beta_2=0.99,
                     layers=[512, 256, 200, 128, 4]):
    hp = {'lamda': lamda,
          'lr': learning_rate,
          'epochs': epochs,
          'batch_size': batch_size,
          'adams_beta': [beta_1, beta_2],
          'layers': layers}

    return hp


def train_dev_test_sets(df_loc='raw_af_data.txt', s_train=0.9, s_dev=0.05,
                        alpha_range=[-5, 15], Re_range=[500000, 10000000],
                        inputs_list=['af', 're', 'a'], outputs_list=['cl', 'cd', 'cdp', 'cm'],
                        normalize=['re', 'a', 'cl', 'cd', 'cdp', 'cm']):
    sets = {}

    # create input
    data = pd.read_csv(df_loc)
    index = data.index
    h = data.copy()
    # modify data
    h = h[(h['a'] > alpha_range[0]) & (h['a'] < alpha_range[1])]
    h = h[(h['re'] > Re_range[0]) & (h['re'] < Re_range[1])]
    # normalize the desired columns
    n_params = {}
    for col in normalize:
        mu = h[col].mean()
        sigma = h[col].std()
        h[col] = (h[col] - mu) / sigma
        n_params[col] = {'mu': mu, 'sigma': sigma}

    # shuffle the data three times
    # shuffle the data three times
    h = h.sample(frac=1, axis=0).reset_index(drop=True)
    h = h.reindex(np.random.permutation(h.index)).reset_index(drop=True)
    ## third shuffle
    inputs_train = h.sample(frac=s_train)
    remaining = h.drop(inputs_train.index)
    inputs_train = inputs_train.reset_index(drop=True)
    inputs_dev = remaining.sample(frac=(s_dev / (1 - s_train)))
    inputs_test = remaining.drop(inputs_dev.index).reset_index(drop=True)
    inputs_dev = inputs_dev.reset_index(drop=True)

    sets['x_train'] = inputs_train[inputs_list].values.transpose()
    sets['y_train'] = inputs_train[outputs_list].values.transpose()
    sets['m_train'] = sets['x_train'].shape[1]

    sets['x_dev'] = inputs_dev[inputs_list].values.transpose()
    sets['y_dev'] = inputs_dev[outputs_list].values.transpose()
    sets['m_dev'] = sets['x_dev'].shape[1]

    sets['x_test'] = inputs_test[inputs_list].values.transpose()
    sets['y_test'] = inputs_test[outputs_list].values.transpose()
    sets['m_test'] = sets['x_test'].shape[1]

    return n_params, sets


def create_network(sets, label_af, af_data_dic, layers=[512, 256, 200, 128, 4], lamda=0.8, seed=0):
    network = {}
    tf.reset_default_graph()
    network['n'] = af_data_dic[label_af[0]]['input'].shape[1] + 2
    network['A0'] = tf.placeholder('float64', shape=(network['n'], None))
    network['y'] = tf.placeholder('float64', shape=(sets['y_train'].shape[0], None))
    a = [network['n']]
    a.extend(layers)
    layers = a
    network['layers'] = layers
    for i in range(1, len(layers)):

        network['W' + str(i)] = tf.get_variable("W" + str(i), (layers[i], layers[i - 1]),
                                                initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                                                dtype=tf.float64)
        network['b' + str(i)] = tf.get_variable("b" + str(i), [layers[i], 1],
                                                initializer=tf.zeros_initializer(), dtype=tf.float64)
        network['Z' + str(i)] = tf.add(tf.matmul(network['W' + str(i)], network['A' + str(i - 1)]),
                                       network['b' + str(i)])
        if i != len(layers) - 1:
            network['A' + str(i)] = tf.nn.leaky_relu(network['Z' + str(i)])
        else:
            network['A' + str(i)] = network['Z' + str(i)]
        if i == 1:
            network['reg'] = tf.nn.l2_loss(network['W' + str(i)])
        else:
            network['reg'] += tf.nn.l2_loss(network['W' + str(i)])

    # calculate cost
    network['cost'] = tf.reduce_sum((network['y'] - network['A' + str(len(layers) - 1)]) ** 2.)
    # compute reqularized loss
    network['cost'] = tf.reduce_mean(network['cost'] + lamda * network['reg'])

    return network


def test_test_set(network, sets, sess, af_data_dic, label_af, results='res.txt', label='test'):
    x_af = sets['x_test'][0, :].astype(int)
    x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
    x_temp = sets['x_test'][1:3, :]
    x_test_n = np.concatenate((x_af, x_temp), axis=0)

    predict = tf.abs(network['y'] - network['A' + str(len(network['layers']) - 1)])
    error = sess.run(predict, feed_dict={network['A0']: x_test_n, network['y']: sets['y_test']})
    fid = open(results, 'a')
    fid.write('{} infinity norm test cl : {}\n'.format(label, error[0, :].max()))
    fid.write('{} infinity norm test cd : {}\n'.format(label, error[1, :].max()))
    fid.write('{} infinity norm test cdp: {}\n'.format(label, error[2, :].max()))
    fid.write('{} infinity norm test cm : {}\n\n'.format(label, error[3, :].max()))
    fid.close()

    return


def test_dev_set(network, sets, sess, af_data_dic, label_af, results='res.txt', label='dev'):

    # create x_dev
    x_af = sets['x_dev'][0, :].astype(int)
    x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
    x_temp = sets['x_dev'][1:3, :]
    x_dex_n = np.concatenate((x_af, x_temp), axis=0)

    # predict the error
    predict = tf.abs(network['y'] - network['A' + str(len(network['layers']) - 1)])
    error = sess.run(predict, feed_dict={network['A0']: x_dex_n, network['y']: sets['y_dev']})
    # print(error.shape)
    # time.sleep(1)
    fid = open(results, 'a')
    fid.write('infinity norm:\n\n')
    fid.write('{} infinity norm cl : {}\n'.format(label, error[0, :].max()))
    fid.write('{} infinity norm cd : {}\n'.format(label, error[1, :].max()))
    fid.write('{} infinity norm cdp: {}\n'.format(label, error[2, :].max()))
    fid.write('{} infinity norm cm : {}\n'.format(label, error[3, :].max()))
    fid.write('{} cl  std {} mean {}\n'.format(label, error[0, :].std(), error[0, :].mean()))
    fid.write('{} cd  std {} mean {}\n'.format(label, error[1, :].std(), error[1, :].mean()))
    fid.write('{} cdp std {} mean {}\n'.format(label, error[2, :].std(), error[2, :].mean()))
    fid.write('{} cm  std {} mean {}\n\n'.format(label, error[3, :].std(), error[3, :].mean()))
    fid.close()

    return


if "__main__" == __name__:

    af_data_dic, af_label, label_af = load_data()
    hp = hyper_parameters()
    n_params, sets = train_dev_test_sets()
    tf.reset_default_graph()
    network = create_network(sets, label_af, af_data_dic, hp['layers'], hp['lamda'])
    optimizer = tf.train.AdamOptimizer(learning_rate=hp['lr'],
                                       beta1=hp['adams_beta'][0],
                                       beta2=hp['adams_beta'][0]).minimize(network['cost'])
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    seed = 0
    costs = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)

        for epoch in range(hp['epochs']):
            epoch_cost = 0.
            n_batch = int(sets['m_train'] / hp['batch_size'])
            seed = seed + 1
            minibatches = random_mini_batches(sets['x_train'], sets['y_train'], hp['batch_size'], seed)
            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch
                # convert the x_s
                x_af = minibatch_x[0, :].astype(int)
                x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
                x_temp = minibatch_x[1:3, :]
                minibatch_x = np.concatenate((x_af, x_temp), axis=0)
                _, minibatch_cost = sess.run([optimizer, network['cost']], feed_dict={network['A0']: minibatch_x,
                                                                                      network['y']: minibatch_y})
                epoch_cost += minibatch_cost / n_batch

            # Print the cost every epoch
            if epoch % 1 == 0:
                fid = open('res.txt', 'a')
                fid.write("Cost after epoch {}: {}\n".format(epoch, epoch_cost))
                fid.close()
            if epoch % 1 == 0:
                costs.append(epoch_cost)
                print(epoch_cost)

            if epoch % 10 == 0:
                cp = saver.save(sess, 'cp/cp.ckpt')
                test_dev_set(network, sets, sess, af_data_dic, label_af, results='res.txt', label='dev')

            if epoch % 50 == 0:
                test_test_set(network, sets, sess, af_data_dic, label_af, results='res.txt', label='test')

