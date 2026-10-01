import numpy as np,sys,json
import ctl
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.39
OPEN=float(sys.argv[1]); CLOSE=float(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
OBJ=int(sys.argv[4]) if len(sys.argv)>4 else 32
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); tg=o[OBJ:OBJ+3].copy(); bb=o[OBJ+13:OBJ+16].copy()
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],tg[1],0.0,steps=20)
for tx in [-0.18,-0.14]: obs=base_goto(env,obs,tx,tg[1],0.0,steps=12,vmax=0.04)
zg=0.46+0.012
obs,u=servo(env,obs,[tg[0],tg[1],zg+0.15],grip=OPEN,max_steps=90)
print("above",u,np.round(ctl.tcp_world(obs),3),"obj",np.round(np.asarray(obs)[OBJ:OBJ+3],3))
obs,u=servo(env,obs,[tg[0],tg[1],zg],grip=OPEN,max_steps=40,chunk=4)
print("down",u,np.round(ctl.tcp_world(obs),3),"obj",np.round(np.asarray(obs)[OBJ:OBJ+3],3))
obs=hold(env,obs,8,CLOSE)
p=ctl.tcp_world(obs)
obs,u=servo(env,obs,p+np.array([0,0,0.12]),grip=CLOSE,max_steps=40,chunk=4)
o=np.asarray(obs)
print("lift",u,"tcp",np.round(ctl.tcp_world(obs),3),"obj",np.round(o[OBJ:OBJ+3],3),"grip",o[103])
json.dump(o.tolist(),open("lift_snap.json","w"))
env.close()
