import subprocess as sp
import os
import shutil
import time
import multiprocessing as mp


re_min = 1000000
re_max = 2000000
num_core = 3


def xfoil(params_list):

    cur_dir = params_list[0]
    move_dir = params_list[1]
    Re = params_list[2]
    af_name = params_list[3]
    alpha_min = params_list[4]
    alpha_max = params_list[5]
    alpha_increment = params_list[6]

    with open(af_name, 'w') as fid2:
        fid2.write(af_name + '\n')
        fid2.write('save {}.af\nY\n'.format(af_name))
        fid2.write('plop\n')
        fid2.write('g\n\noper\niter 2000\n')
        fid2.write('visc {}\n'.format(str(Re)))
        fid2.write('pacc\n\n\n')
        fid2.write('aseq {} {} {}\n'.format(alpha_min, alpha_max, alpha_increment))
        fid2.write('pacc\npwrt\n')
        fid2.write(af_name + '_Re_' + str(Re) + '.polar\nY\n\n\nquit\n')

    # commands to be executed
    fid = open(cur_dir + 'missed.txt', 'a')
    try:
        t = time.time()
        xx = sp.Popen(cur_dir + 'xfoil < {}'.format(af_name), shell=True, stdout=False)
        xx.wait()
        time.sleep(10)
        #xx.kill()
        #shutil.move(cur_dir + '{}'.format(af_name), move_dir)
        os.remove(cur_dir + '{}'.format(af_name))
        shutil.move(cur_dir + '{}'.format(af_name + '_Re_' + str(Re) + '.polar'), move_dir)
        shutil.move(cur_dir + '{}'.format(af_name + '.af'), move_dir)
        th = time.time() - t
        fid.write(str(th) + '\n')
    except:
        fid.write(af_name + '\n')

    fid.close()
    return


def create_data_structure(Re=[400000, 1000000, 50000],
                          af_camber=[0, 16],
                          af_thickness=[12, 36],
                          alpha_range=[-10, 35, 1.]):

    data_structure = []
    cur_dir = os.getcwd() + '/'
    Re = range(Re[0], Re[1], Re[2])
    af_camber = range(af_camber[0], af_camber[1])
    af_thickness = range(af_thickness[0], af_thickness[1])

    for i in Re:
        if not os.path.isdir(cur_dir + '{}/'.format(i)):
            os.mkdir(cur_dir + '{}/'.format(i))
        xfoil_address = cur_dir + '{}/'.format(i)
        for j in af_camber:
            for k in af_thickness:

                if len(str(j)) == 1:
                    first_string = '0' + str(j)
                else:
                    first_string = str(j)

                if len(str(k)) == 1:
                    second_string = '0' + str(k)
                else:
                    second_string = str(k)
                af_name = 'naca' + first_string + second_string

                data_structure.append([cur_dir, xfoil_address, i, af_name,
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
