import numpy as np,sys,json
from env_client import make_env
from ctl import *
seed=int(sys.argv[1]); OBJ=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); tg=o[OBJ:OBJ+3].copy(); bb=o[OBJ+13:OBJ+16].copy()
print("obj",np.round(tg,3),"bb",np.round(bb,3))
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],tg[1],0.0,steps=20)
for tx in [-0.18,-0.14]:
    obs=base_goto(env,obs,tx,tg[1],0.0,steps=12,vmax=0.04)
print("base",np.round(np.asarray(obs)[93:96],3))
gz=tg[2]+bb[2]*0.5-0.03
tgt=np.array([tg[0],tg[1],gz])
obs,u=servo(env,obs,tgt+np.array([0,0,0.12]),max_steps=90)
print("above u",u,np.round(tcp_world(obs),3),"obj",np.round(np.asarray(obs)[OBJ:OBJ+3],3))
obs,u=servo(env,obs,tgt,max_steps=40,chunk=4)
print("down u",u,np.round(tcp_world(obs),3),"obj",np.round(np.asarray(obs)[OBJ:OBJ+3],3))
obs=hold(env,obs,6,1.0)
p=tcp_world(obs)
obs,u=servo(env,obs,p+np.array([0,0,0.12]),grip=1.0,max_steps=40,chunk=4)
o=np.asarray(obs)
print("lift",u,"tcp",np.round(tcp_world(obs),3),"obj",np.round(o[OBJ:OBJ+3],3))
json.dump(o.tolist(),open("lift_snap.json","w"))
env.close()
