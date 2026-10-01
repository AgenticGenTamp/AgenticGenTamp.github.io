import numpy as np,sys
import ctl
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.44
YAW=-1.5708; DX,DY=0.019,0.17   # real = model + (DX,DY) at this yaw
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); can=o[32:35].copy(); bowl=o[0:3].copy()
print("can",np.round(can,3),"bowl",np.round(bowl,3))
start=np.array([can[0]-DX, can[1]-DY-0.07, 0.53])
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],start[1],0.0,steps=25)
for tx in [-0.18,-0.145]: obs=base_goto(env,obs,tx,start[1],0.0,steps=12,vmax=0.04)
obs,u=servo(env,obs,start+np.array([0,0,0.12]),grip=0.0,max_steps=80,yaw=YAW)
obs,u=servo(env,obs,start,grip=0.0,max_steps=30,chunk=5,yaw=YAW)
print("start tcp",np.round(ctl.tcp_world(obs),3),"can",np.round(np.asarray(obs)[32:35],3))
y=start[1]
for i in range(24):
    y+=0.02
    bo=np.asarray(obs)
    # keep base aligned
    if abs(bo[94]-y)>0.10: obs=base_goto(env,obs,bo[93],y,0.0,steps=6,vmax=0.06)
    obs,u=servo(env,obs,[start[0],y,0.53],grip=0.0,max_steps=14,chunk=4,yaw=YAW)
    o=np.asarray(obs)
    a=np.zeros(11,dtype=np.float32); obs,r,te,tr,inf=env.step(a)
    print("y=%.3f tcp=%s can=%s r=%.3f term=%s"%(y,np.round(ctl.tcp_world(obs),3),np.round(np.asarray(obs)[32:35],3),r,te))
    if te: break
env.close()
