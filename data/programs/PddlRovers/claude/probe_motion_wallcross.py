from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def setth(obs,th,i=0):
    for _ in range(60):
        d=th-pose(obs,i)[2]
        if abs(d)<0.0005: break
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),i)
    return obs
obs=setth(obs,0.0,0)
for y in [-2.27,-2.25,-2.3,2.25,2.27,2.0]:
    obs,ok=goto(env,obs,0.6,y,0); p=pose(obs,0)
    if abs(p[0]-0.6)>0.02 or abs(p[1]-y)>0.02:
        print("y=%.3f goto failed at %s"%(y,np.round(p,3))); continue
    obs,pp=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005)
    print("y=%6.3f (actual %.3f) west-blocked x=%.5f"%(y,pp[1],pp[0]))
# also probe wall from east at fine y grid to find any gap
obs,ok=goto(env,obs,0.6,-1.0,0)
for y in np.arange(-2.25,2.3,0.25):
    obs,ok=goto(env,obs,0.6,float(y),0); p=pose(obs,0)
    if abs(p[0]-0.6)>0.02 or abs(p[1]-y)>0.02: print("y=%5.2f skip(%s)"%(y,np.round(p,2))); continue
    obs,pp=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.0005)
    print("y=%5.2f xmin=%.4f"%(y,pp[0]))
    obs,ok=goto(env,obs,0.6,float(y),0)
env.close()
