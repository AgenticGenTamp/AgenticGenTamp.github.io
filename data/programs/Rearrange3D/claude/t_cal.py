import numpy as np,sys
import ctl
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.44
mode=sys.argv[1]
YAW=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
env=make_env(); obs,info=env.reset(seed=0)
o=np.asarray(obs); OBJ=32
def can(obs): return np.asarray(obs)[OBJ:OBJ+3].copy()
c=can(obs); print("can",np.round(c,3))
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],c[1],0.0,steps=20)
for tx in [-0.18,-0.14]: obs=base_goto(env,obs,tx,c[1],0.0,steps=12,vmax=0.04)
Z=0.53
def sweep(obs,start,d,n=26):
    obs,u=servo(env,obs,start+np.array([0,0,0.12]),grip=0.0,max_steps=80,yaw=YAW)
    obs,u=servo(env,obs,start,grip=0.0,max_steps=30,chunk=5,yaw=YAW)
    ref=can(obs); print("  start tcp",np.round(ctl.tcp_world(obs),3))
    for i in range(n):
        p=start+d*(i+1)
        obs,u=servo(env,obs,p,grip=0.0,max_steps=14,chunk=4,yaw=YAW)
        t=ctl.tcp_world(obs); mv=np.linalg.norm(can(obs)-ref)
        if mv>0.004:
            print("  CONTACT at tcp",np.round(t,3),"cmd",np.round(p,3),"mv %.3f"%mv); return obs,t
        print("   tcp",np.round(t,3),"mv %.4f"%mv)
    return obs,None
c=can(obs)
if mode=='ym':
    obs,t=sweep(obs,np.array([c[0],c[1]-0.16,Z]),np.array([0,0.01,0]),n=40)
elif mode=='yp':
    obs,t=sweep(obs,np.array([c[0],c[1]+0.16,Z]),np.array([0,-0.01,0]))
elif mode=='xm':
    obs,t=sweep(obs,np.array([c[0]-0.18,c[1],Z]),np.array([0.01,0,0]))
elif mode=='ymr':
    obs,t=sweep(obs,np.array([c[0],c[1]-0.16,Z]),np.array([0,0.01,0]),n=40)
env.close()
