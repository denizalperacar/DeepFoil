import subprocess as sp
import os
import shutil
import time
import multiprocessing as mp
from glob import glob



re_min = 1000000
re_max = 999990
num_core = 2


def xfoil(params_list):

    cur_dir = params_list[0]
    move_dir = params_list[1]
    Re = params_list[2]
    af_name = params_list[3]
    alpha_min = params_list[4]
    alpha_max = params_list[5]
    alpha_increment = params_list[6]

    with open(af_name[:-3]+'.run', 'w') as fid2:
        fid2.write('load ' + af_name + '\n')
        #fid2.write('save {}.af\nY\n'.format(af_name))
        fid2.write('pane\n\n')
        fid2.write('plop\n')
        fid2.write('g\n\noper\niter 1000\n')
        fid2.write('visc {}\n'.format(str(Re)))
        fid2.write('pacc\n\n\n')
        fid2.write('aseq {} {} {}\n'.format(alpha_min, alpha_max, alpha_increment))
        fid2.write('pacc\npwrt\n')
        fid2.write(af_name[:-3] + '_Re_' + str(Re) + '.polar\n\n\nquit\n')

    # commands to be executed
    fid = open(cur_dir + 'missed.txt', 'a')
    t = time.time()
    try:

        xx = sp.Popen(cur_dir + 'xfoil < {}'.format(af_name[:-3]+'.run'), shell=True, stdout=False)
        xx.wait()
    except:
        fid.write(af_name + '\n')
    time.sleep(20)
    th = time.time() - t
    fid.write(str(th) + '\n')

    shutil.move(cur_dir + '{}'.format(af_name[:-3]+'.run'), move_dir)
        #os.remove(cur_dir + '{}'.format(af_name))
    shutil.move(cur_dir + '{}'.format(af_name[:-3] + '_Re_' + str(Re) + '.polar'), move_dir)
        #shutil.copy(cur_dir + '{}'.format(af_name), move_dir)


    fid.close()
    return


def create_data_structure(Re=[1000000, 1000500, 50000],
                          alpha_range=[-10, 35, 1.]):

    data_structure = []
    cur_dir = os.getcwd() + '/'
    Re = range(Re[0], Re[1], Re[2])

    files = glob(cur_dir+'/*.af')

    for i in Re:
        if not os.path.isdir(cur_dir + '{}/'.format(i)):
            os.mkdir(cur_dir + '{}/'.format(i))
        xfoil_address = cur_dir + '{}/'.format(i)
        for j in files:
            data_structure.append([cur_dir, xfoil_address, i, j.replace(cur_dir, ''),
                                   alpha_range[0], alpha_range[1], alpha_range[2]])

    return data_structure


core_num = num_core
data = create_data_structure(Re=[re_min, re_max + 50000, 50000])
#for i in data:
#    xfoil(i)

p = mp.Pool(processes=core_num)
p.imap(xfoil, data)
p.close()
p.join()
