import numpy as np,sys,json
from env_client import make_env
from ctl import *
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
OBJ=int(sys.argv[2]) if len(sys.argv)>2 else 32
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); tg=o[OBJ:OBJ+3].copy(); bb=o[OBJ+13:OBJ+16].copy()
print("obj",np.round(tg,3),"bb",np.round(bb,3))
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],tg[1],0.0,steps=20)
for tx in [-0.18,-0.14,-0.10]:
    obs=base_goto(env,obs,tx,tg[1],0.0,steps=12,vmax=0.04)
o=np.asarray(obs); print("base",np.round(o[93:96],3),"tcp",np.round(tcp_world(obs),3))
gz=tg[2]+bb[2]*0.5-0.03   # near top of object
tgt=np.array([tg[0],tg[1],gz])
obs,ok=move_world(env,obs,tgt+np.array([0,0,0.15]),steps=60,grip=0.0,maxjump=99)
print("above",ok,np.round(tcp_world(obs),3),"obj",np.round(np.asarray(obs)[OBJ:OBJ+3],3))
obs,oks=move_world_line(env,obs,tcp_world(obs),tgt,n=3,steps_per=8,grip=0.0)
print("down",oks,np.round(tcp_world(obs),3),"obj",np.round(np.asarray(obs)[OBJ:OBJ+3],3))
obs=hold(env,obs,6,1.0)
p=tcp_world(obs)
obs,oks=move_world_line(env,obs,p,p+np.array([0,0,0.15]),n=3,steps_per=8,grip=1.0)
o=np.asarray(obs)
print("lift",oks,"tcp",np.round(tcp_world(obs),3),"obj",np.round(o[OBJ:OBJ+3],3))
json.dump(o.tolist(),open("lift_snap.json","w"))
env.close()
