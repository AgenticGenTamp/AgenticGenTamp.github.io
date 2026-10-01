import numpy as np,sys
import ctl
from ctl import *
from env_client import make_env
ctl.Z_OFF=0.44
DX=0.04
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
relz=float(sys.argv[2]) if len(sys.argv)>2 else 0.52
env=make_env(); obs,info=env.reset(seed=seed)
o=np.asarray(obs); can=o[32:35].copy(); bowl=o[0:3].copy()
print("can",np.round(can,3),"bowl",np.round(bowl,3))
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
obs=base_goto(env,obs,np.asarray(obs)[93],can[1]-0.12,0.0,steps=30)
obs=base_goto(env,obs,-0.20,can[1]-0.12,0.0,steps=12,vmax=0.05)
o=np.asarray(obs)
tgt=np.array([can[0]-DX, o[94], relz])
obs,u=servo(env,obs,tgt+np.array([0,0,0.1]),grip=0.0,max_steps=70)
obs,u=servo(env,obs,tgt,grip=0.0,max_steps=25,chunk=5)
qhold=np.asarray(obs)[96:103].copy()
print("modelTCP",np.round(ctl.tcp_world(obs),3),"base",np.round(np.asarray(obs)[93:96],3))
for i in range(46):
    a=np.zeros(11,dtype=np.float32)
    q=np.asarray(obs)[96:103]
    a[3:10]=np.clip(3.0*(qhold-q),-0.1,0.1)
    a[1]=0.0115
    obs,r,te,tr,inf=env.step(a)
    o=np.asarray(obs)
    if i%3==0 or r!=-1.0 or te:
        print("i%d by=%.3f can=%s r=%.4f t=%s"%(i,o[94],np.round(o[32:35],3),r,te))
    if te: break
env.close()
