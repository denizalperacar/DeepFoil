import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import factorial, tan, pi
from glob import glob
import os
import json
from matplotlib.pyplot import figure

def binomial(r, n):
    return factorial(n) / (factorial(r) * factorial(n-r))

def bpn(x, r, n):
    x = np.array(x)
    results = binomial(r,n) * (x**r) * ((1-x)**(n-r))
    return results
def show_bpn(x, n):
    for r in range(n+1):
        plt.plot(x, bpn(x, r, n))
    plt.show()
    return
def get_class(x, n1, n2):
    return (x ** n1) * ((1-x)**n2)
def show_af_bpn(x, n1, n2, n):
    c = get_class(x, n1, n2)
    for r in range(n+1):
        plt.plot(x, bpn(x, r, n) * c)
    plt.show()
    return

def airfoil(Nnose = 0.5,
            Naft  = 1.0,
            n     = 4,
            x     = np.concatenate((np.linspace(0, 0.195, 81), np.linspace(0.2, 1., 101))) ,
            rle   = 0.01,
            bu    = 20.,
            dzu   = 0.002,
            bl    = 20.,
            dzl   = -0.001,
            au_lim = [-0.1, 0.6],
            al_lim = [-0.6, 0.1], show=True):

    au    = np.random.uniform(au_lim[0], au_lim[1], n+1)
    al    = np.random.uniform(al_lim[0], al_lim[1], n+1)
    au[0] = (rle *2)**0.5
    au[-1]= tan(bu * pi / 180.) + dzu
    al[0] = -(rle *2)**0.5
    al[-1]= tan(bl * pi / 180.) + dzl

    # the calculation
    c = get_class(x, Nnose, Naft)
    # create upper surface
    # calculate Si
    resu = np.zeros_like(x)
    resl = np.zeros_like(x)

    for i in range(0, n+1):
        resu += au[i] * bpn(x, i, n)
        resl += al[i] * bpn(x, i, n)

    resu = resu * c + x * dzu
    resl = resl * c + x * dzl

    data_structure = {}
    data_structure['x'] = x
    data_structure['yu'] = resu
    data_structure['yl'] = resl
    data_structure['au'] = au
    data_structure['al'] = al
    data_structure['Nnose'] = Nnose
    data_structure['Naft'] = Naft
    data_structure['rle'] = rle
    data_structure['b'] = [bu, bl]
    data_structure['dz'] = [dzu, dzl]
    data_structure['n'] = n
    if show:
        fig, ax = plt.subplots()
        ax.plot(x, resu)
        ax.plot(x, resl)
        ax.axis('equal')
        fig.show()
    return data_structure

def create_file(af, plot_af=False, directory='afs/'):
    # add directory and the ability to save the ones we want
    os.makedirs(directory, exist_ok = True)
    os.makedirs(directory+'/plot/', exist_ok = True)
    os.makedirs(directory+'/airfoil/', exist_ok = True)
    os.makedirs(directory+'/pickle/', exist_ok = True)

    for i in af.keys():
        x = af[i]['x']
        yu = af[i]['yu']
        yl = af[i]['yl']

        with open(directory + '/airfoil/af_{}.af'.format(i), 'w') as fid:
            fid.write('index {}\n'.format(i))
            for pt in range(x.shape[0]):
                fid.write('{:6.5f} {:6.5f}\n'.format(x[x.shape[0]-pt-1], yu[x.shape[0]-pt-1]))
            for pt in range(1,x.shape[0]):
                fid.write('{:6.5f} {:6.5f}\n'.format(x[pt], yl[pt]))

        with open(directory + '/pickle//af_{}.pickle'.format(i), 'wb') as fid:
            pickle.dump(af[i], fid)

        if plot_af:
            fig, ax = plt.subplots()
            ax.plot(x, yu)
            ax.plot(x, yl)
            ax.axis('equal')
            plt.title(i)
            plt.grid()
            fig.savefig(directory + '/plot/af_{}.png'.format(i), dpi=100)
            plt.close(fig)
            fig, ax = [],[]
    return


if not os.path.exists('index.txt'):
    num = 0
else:
    with open('index.txt', 'r') as fid:
        num = int(fid.read())

cd = os.getcwd() + '/'

for j in range(20):
    newafs = 50000
    num = newafs * j + num
    for i in range(num, num + newafs):
        af = {}
        n=np.random.randint(2,8)
        rle=np.random.uniform(0.01,0.05)
        bu=np.random.randint(-10,20)
        bl=np.random.randint(-20,bu)
        dzu = np.random.uniform(-0.001, 0.005)
        dzl = np.random.uniform(-0.005, dzu-0.001)
        Nnose = np.random.uniform(0.45,0.55)
        Naft  = np.random.uniform(0.95, 1.0)

        af[i] = airfoil(show=False, n=n, rle=rle, bu=bu, bl=bl, dzl=dzl, dzu=dzu, Naft=Naft, Nnose=Nnose,
                        x=np.concatenate((np.linspace(0, 0.195, 61), np.linspace(0.2, 1., 101))))

        directory='{}/afs/af_{}'.format(cd, j)
        os.makedirs(directory, exist_ok = True)

        create_file(af, plot_af=False, directory=directory)
        with open('index.txt', 'w') as fid:
            fid.write(str(i))

        if (i - num) % 1000 == 0:
            print((i-num)/1000)
