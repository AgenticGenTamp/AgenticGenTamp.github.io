import numpy as np,sys
import ctl
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.44
axis=sys.argv[1]; zt=float(sys.argv[2]) if len(sys.argv)>2 else 0.52
env=make_env(); obs,info=env.reset(seed=0)
o=np.asarray(obs); bowl=o[0:3].copy(); print("bowl",np.round(bowl,3))
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],bowl[1],0.0,steps=20)
for tx in [-0.18,-0.14]: obs=base_goto(env,obs,tx,bowl[1],0.0,steps=12,vmax=0.04)
if axis=='x': start=np.array([bowl[0]-0.30,bowl[1],zt]); d=np.array([0.015,0,0])
else: start=np.array([bowl[0],bowl[1]-0.30,zt]); d=np.array([0,0.015,0])
obs,u=servo(env,obs,start+np.array([0,0,0.1]),grip=0.0,max_steps=90)
obs,u=servo(env,obs,start,grip=0.0,max_steps=30,chunk=5)
print("start tcp",np.round(ctl.tcp_world(obs),3))
ref=np.asarray(obs)[0:3].copy()
for i in range(28):
    p=start+d*(i+1)
    obs,u=servo(env,obs,p,grip=0.0,max_steps=12,chunk=4)
    o=np.asarray(obs); mv=np.linalg.norm(o[0:3]-ref)
    print("cmd %s tcp %s bowlmove %.4f"%(np.round(p,3),np.round(ctl.tcp_world(obs),3),mv))
    if mv>0.004: print("CONTACT"); break
env.close()
