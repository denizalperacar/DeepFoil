import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import math
import time

class AirfoilPredictor(torch.nn.Module):
    def __init__(self):
        super(AirfoilPredictor, self).__init__()
        self.net = {}
        self.d = 326
        self.lamda = 0.00001
        self.eta = 0.00001
        self.beta_1 = 0.95
        self.beta_2 = 0.999
        self.ind = 0
        self.inner = 300

        self.l1  = torch.nn.Linear(self.d, self.inner, bias=True)
        self.l2  = torch.nn.Linear(self.inner, self.inner, bias=True)
        self.l3  = torch.nn.Linear(self.inner, self.inner, bias=True)
        self.l4  = torch.nn.Linear(self.inner, self.inner, bias=True)
        self.l5  = torch.nn.Linear(self.inner, self.inner, bias=True)
        self.l6  = torch.nn.Linear(self.inner, 1, bias=True)

        # self.r5  = torch.nn.Linear(350, 300, bias=True)
        # self.r6  = torch.nn.Linear(300, 150, bias=True)

        self.a = torch.nn.LeakyReLU() #negative_slope=0.9)
    def forward(self, x):

        x   = self.a(self.l1(x))
        x   = self.a(self.l2(x))
        x   = self.a(self.l3(x))
        x   = self.a(self.l4(x))
        x   = self.a(self.l5(x))
        x   = self.l6(x)

        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print(device)
model = AirfoilPredictor().to(device)
print(model)
loss_f = torch.nn.MSELoss() #reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = model.eta, betas=[model.beta_1, model.beta_2])

# load_mode;
model.load_state_dict(torch.load('weights/net_1'))

def load_data(af_points='tr_af_points.pickle',
              af_labels='tr_af_labels.pickle',
              label_afs='tr_label_afs.pickle'):

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


def train_dev_test_sets(df_loc='tr_raw_af_data', s_train=0.99,
                        alpha_range=[-5, 20], Re_range=[900000, 10000000],
                        inputs_list=['af', 're', 'a'], outputs_list=['cd'],
                        normalize=['re', 'a', 'cl', 'cd', 'cdp', 'cm']):
    sets = {}

    # create input
    data_train = pd.read_csv(df_loc+'.txt')
    index = data_train.index
    h = data_train.copy()
    # modify data
    h = h[(h['a'] > alpha_range[0]) & (h['a'] < alpha_range[1])]
    h = h[(h['re'] > Re_range[0]) & (h['re'] < Re_range[1])]
    # normalize the desired columns
    n_params = {}

    with open('norms:\n', 'w') as fid:
        fid.write('\n')

    #for col in normalize:
    #    mu = h[col].mean()
    #    sigma = h[col].std()
    #    h[col] = (h[col] - mu) / sigma
    #    n_params[col] = {'mu': mu, 'sigma': sigma}
    #    with open('norms.txt', 'a') as fid:
    #        fid.write(col+'\n'+str(mu)+' '+str(sigma))

    h['re'] = h['re'] / 8000000.
    h['a'] = h['a'] / 30.

    # lets make the outputs bigger so that we decrease the error more rapidly and make the error smaller 10.11.19
    ###
    h['cl'] = (h['cl'] / 4.8) + 0.5
    h['cd'] = (h['cd']) / 0.2
    h['cdp'] = (h['cdp']) / 0.2
    h['cm'] = (h['cm'] / 0.8) + 0.5

    #for nnn in ['cl', 'cd', 'cdp', 'cm']:
    #    print(nnn, h[nnn].min(), h[nnn].max())
    ###

    # shuffle the data three times
    # shuffle the data three times
    h = h.sample(frac=1, axis=0).reset_index(drop=True)
    h = h.reindex(np.random.permutation(h.index)).reset_index(drop=True)
    ## third shuffle
    inputs_train = h.sample(frac=s_train)
    remaining = h.drop(inputs_train.index)
    inputs_train = inputs_train.reset_index(drop=True)
    inputs_dev = remaining.sample(frac=(1 - s_train)).reset_index(drop=True)

    # create the train and dev sets
    sets['x_train'] = inputs_train[inputs_list].values.transpose()
    sets['y_train'] = inputs_train[outputs_list].values.transpose()
    sets['m_train'] = sets['x_train'].shape[1]

    return n_params, sets


# load everything
af_data_dic, af_label, label_af = load_data()
n_params, sets = train_dev_test_sets()

seed = 0
costs = []
b_s = 5
num_epochs = 2000
for epoch in range(num_epochs):
    t = time.time()
    epoch_cost = 0.
    if (epoch % 5) == 0 and epoch !=0 and b_s < 1000:
        b_s *= 2
        print('nbatch = %f' %(n_batch))
    n_batch = int(sets['m_train']/b_s)

    #    model.eta = 0.1
    # model.eta *= math.exp(-epoch/50)
    optimizer = torch.optim.Adam(model.parameters(), lr = model.eta, betas=[model.beta_1, model.beta_2], amsgrad=True)
    # print('nbatch = %f' %(n_batch))
    seed = seed + 1
    minibatches = random_mini_batches(sets['x_train'], sets['y_train'], b_s, seed)
    # offload the n params and sets
    for minibatch in minibatches:
        (minibatch_x, minibatch_y) = minibatch
        # convert the x_s
        x_af = minibatch_x[0, :].astype(int)
        x_af = np.array([af_data_dic[label_af[i]]['input'].flatten() for i in x_af]).transpose()
        x_temp = minibatch_x[1:3, :]
        minibatch_x = np.concatenate((x_af, x_temp), axis=0)
        minibatch_x = torch.from_numpy(minibatch_x.T).float().to(device)
        minibatch_y = torch.from_numpy(minibatch_y.T).float().to(device)
        optimizer.zero_grad()
        output = model(minibatch_x)
        loss = loss_f(output, minibatch_y)
        minibatch_cost = loss.tolist()
        loss.backward()
        optimizer.step()

        epoch_cost += minibatch_cost / n_batch

    print('Epoch [{:3d}/{:3d}], Loss: {:30.10f} bachsize: {:d}, time {:f}, {:d}'.format(epoch+1, num_epochs, epoch_cost, b_s, (time.time()-t), n_batch))
    #if (epoch % 10) == 0 or epoch == num_epochs-1:
    torch.save(model.state_dict(),'weights/net_{}'.format(epoch+1))
