import numpy as np,sys,json
from env_client import make_env
import ctl
from ctl import *
ctl.Z_OFF=float(sys.argv[1]) if len(sys.argv)>1 else 0.44
seed=0; OBJ=32
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); tg=o[OBJ:OBJ+3].copy()
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],tg[1],0.0,steps=20)
for tx in [-0.18,-0.14]: obs=base_goto(env,obs,tx,tg[1],0.0,steps=12,vmax=0.04)
obs,u=servo(env,obs,[tg[0],tg[1],0.64],max_steps=90)
print("start",np.round(tcp_world(obs),3))
for z in np.arange(0.62,0.42,-0.015):
    obs,u=servo(env,obs,[tg[0],tg[1],z],max_steps=16,chunk=4)
    o=np.asarray(obs)
    print("z=%.2f tcp=%s obj=%s"%(z,np.round(tcp_world(obs),3),np.round(o[OBJ:OBJ+3],3)))
env.close()
