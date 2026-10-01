import numpy as np,sys,json
from env_client import make_env
from ctrl import *
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0)
o=np.asarray(obs); base=o[93:96].copy(); can=o[32:35].copy(); drink=o[16:19].copy()
print("tcp0",np.round(tcp_world(obs),3),"can",np.round(can,3))
tgt=np.array([can[0],can[1],can[2]+0.02])
obs,ok=move_world(env,obs,tgt+np.array([0,0,0.18]),steps=80,grip=0.0,maxjump=99)
print("above ok",ok,np.round(tcp_world(obs),3),"can",np.round(np.asarray(obs)[32:35],3))
obs,oks=move_world_line(env,obs,tcp_world(obs),tgt,n=4,steps_per=10,grip=0.0)
print("down",oks,np.round(tcp_world(obs),3),"can",np.round(np.asarray(obs)[32:35],3))
for i in range(8):
    a=np.zeros(11,dtype=np.float32);a[10]=1.0;obs,r,te,tr,inf=env.step(a)
print("closed can",np.round(np.asarray(obs)[32:35],3),"r",r)
p=tcp_world(obs)
obs,oks=move_world_line(env,obs,p,p+np.array([0,0,0.2]),n=4,steps_per=10,grip=1.0)
o=np.asarray(obs)
print("lift",oks,"tcp",np.round(tcp_world(obs),3),"can",np.round(o[32:35],3),"r",r)
json.dump(o.tolist(),open("lift_snap.json","w"))
env.close()
