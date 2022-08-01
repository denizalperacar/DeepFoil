import subprocess as sp
import shutil
from time import time, sleep
from threading import Timer
sys.path.append("/opt/lib/python2.6/site-packages")
#/opt/lib/python2.6/site-packages
import psutil


def get_dirs(wd):
    rd = glob(wd+'/*')
    r = {}
    for i in rd:
        r[i.replace(wd, '')] = {}
        r[i.replace(wd, '')]['runs'] = [j.replace(i, '') for j in glob(i + '/*.run')]
        r[i.replace(wd, '')]['starts'] = [j.replace(i, '') for j in glob(i + '/*.start')]
        r[i.replace(wd, '')]['xf'] = [j.replace(i, '') for j in glob(i + '/*.start')]
    return r

def make_ds(wd):
    r = get_dirs(wd)
    ds = []
    k = 0
    for i in r.keys():
        for j in r[i]['runs']:
            if j[:-4]+'.start' not in r[i]['starts']:
                ds.append([i, j, k])
                k +=1
    #print(ds)
    return ds


def kill_porc(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        print(proc.pid)
        proc.kill()
    process.kill()
    return

def runs(oper):
    #print(get_dirs(wd))
    try:
        af = oper[0]
        rf = oper[1]
        kk = oper[2]

        if kk % 200 == 0 and kk != 0:
            # delete the extra xfoil files
            h = get_dirs(wd)
            for fold in h.keys():
                if (len(h[fold]['runs']) == len(h[fold]['runs'])) and os.path.exists(wd + '/' + fold):
                    try:
                        os.remove(wd + '/' + fold + '/xfoil')
                        print('deleted.')
                    except:
                        varr = 0

        #cwd = '{}/{}/'.format(wd, af).replace('//', '/')
        cwd = wd + '/' + af + '/'
        cwd.replace('//', '/')

        if not os.path.exists(cwd + '/xfoil'):
            h1 = cwd.replace('//', '/').replace('/runs/' + af + '/', '')
            shutil.copy(h1 + '/xfoil', cwd)

        #cmd = 'cd {}; xfoil < {}'.format(cwd,rf[1:]).replace('//', '/')
        cmd = 'cd ' + cwd + '; xfoil < ' + rf[1:]
        cmd = cmd.replace('//', '/')
        print(cmd)

        h2 = cwd + '/' + rf[:-4] + '.start'
        with open(h2, 'w') as fidd:
            fidd.write(str(time()))
        kill = lambda process:process.kill()

        prc = sp.Popen(cmd, preexec_fn=os.setsid, shell=True)
        try:
            prc.wait(60)
        except:
            kill_porc(prc.pid)

    except:
        #print(oper)
        a = 1
    return

core_num = 60
cd = '/cfd/kourosh/xfoil_main/'
#wd = '{}/xf/runs/'.format(cd).replace('//', '/')
wd = cd + '/xf/runs/'
wd = wd.replace('//', '/')
print(wd)

ds = make_ds(wd)
#print(ds[:10])
p = mp.Pool(processes=core_num)
p.imap(runs, ds)
p.close()
p.join()
                                                                                                                                                                                                                      99,1          Bot
