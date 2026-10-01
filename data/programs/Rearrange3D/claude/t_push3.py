import numpy as np,sys
import ctl
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.44
DX=0.04
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); dr=o[16:19].copy(); bowl=o[0:3].copy(); can=o[32:35].copy()
print("drink",np.round(dr,3),"bowl",np.round(bowl,3),"d0=%.3f"%np.linalg.norm(dr[:2]-bowl[:2]))
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],dr[1]+0.12,0.0,steps=30)
obs=base_goto(env,obs,-0.20,dr[1]+0.12,0.0,steps=12,vmax=0.05)
o=np.asarray(obs)
tgt=np.array([dr[0]-DX, o[94], 0.51])
obs,u=servo(env,obs,tgt+np.array([0,0,0.1]),grip=0.0,max_steps=70)
obs,u=servo(env,obs,tgt,grip=0.0,max_steps=25,chunk=5)
qhold=np.asarray(obs)[96:103].copy()
print("modelTCP",np.round(ctl.tcp_world(obs),3),"drink",np.round(np.asarray(obs)[16:19],3))
for i in range(46):
    a=np.zeros(11,dtype=np.float32)
    q=np.asarray(obs)[96:103]
    a[3:10]=np.clip(3.0*(qhold-q),-0.1,0.1)
    a[1]=-0.0115
    obs,r,te,tr,inf=env.step(a)
    o=np.asarray(obs)
    d=np.linalg.norm(o[16:18]-o[0:2])
    if i%3==0 or te: print("i%d by=%.3f drink=%s d=%.3f r=%.3f t=%s"%(i,o[94],np.round(o[16:19],3),d,r,te))
    if te: break
env.close()
