import numpy as np, sys
from multiprocessing import Pool
from calib import run
from env_client import make_env
if __name__=='__main__':
    seed=int(sys.argv[1])
    env=make_env(); o,_=env.reset(seed=seed); env.close()
    print('bb bowl',o[13:16],'drink',o[29:32],'can',o[45:48])
    jobs=[]
    for r in (0.135,0.145,0.155,0.165):
        for a in (-90,-45,0):
            ang=np.radians(a); jobs.append((seed,(0.0,0.10),(r*np.cos(ang),r*np.sin(ang))))
    for r in (0.135,0.145,0.155):
        for a in (90,45):
            ang=np.radians(a); jobs.append((seed,(r*np.cos(ang),r*np.sin(ang)),(0.0,-0.10)))
    with Pool(12) as p:
        for x in p.map(run,jobs): print(x[1:5], x[5], x[6], np.linalg.norm(x[5]).round(3), np.linalg.norm(x[6]).round(3))
