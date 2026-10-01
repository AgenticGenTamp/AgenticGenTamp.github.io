from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
for th in (1.25, -1.9):
    obs,ok=nav(env,obs,O[0],O[1]-2.05,i=0)
    pp=pose(obs,0); dth=th-pp[2]
    while abs(dth)>0.02:
        obs,_,_,_,_=st(env,dth=float(np.clip(dth,-0.4,0.4)),i=0); dth=th-pose(obs,0)[2]
    for k in range(25):
        p=pose(obs,0); d=np.linalg.norm(p[:2]-O)
        obs,v=vis_test(env,obs,0)
        if v:
            print("theta=%.2f flip at d=%.4f"%(p[2],d)); break
        obs,_,_,_,_=st(env,dy=0.01,i=0)
        if abs(pose(obs,0)[1]-p[1])<1e-9: print("stuck"); break
env.close()
