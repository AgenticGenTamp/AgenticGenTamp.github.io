import json
import numpy as np
from env_client import make_env
from kinova import planar_ik,fk

env=make_env();s,_=env.reset(seed=0)
def move(b,q,g,n,step=.04):
 global s
 for i in range(n):
  a=np.zeros(11);a[:3]=np.clip(np.asarray(b)-s[125:128],-step,step)
  a[3:10]=np.clip(q-s[128:135],-.1,.1);a[10]=g
  s,r,t,tr,inf=env.step(np.clip(a,env.action_space.low,env.action_space.high))

def log(name):
 print(name,'drawer',round(float(s[107]),3),'cubes',s[:80].reshape(5,16)[:,:3].round(3).tolist(),'base',s[125:128].round(3).tolist(),'fk',fk(s[128:135])[:3,3].round(3).tolist(),flush=True)
 np.save('cube_place_'+name+'.npy',s)
qdrawer=planar_ik(.6,-.05,-np.pi/2);qdrawer[6]+=np.pi/2
move([2.,0.,np.pi],qdrawer,0,150)
move([1.6,0.,np.pi],qdrawer,0,45)
move([1.6,0.,np.pi],qdrawer,1,15)
move([2.1,0.,np.pi],qdrawer,1,50,.03)
log('drawer')
qhigh=planar_ik(.7,.2);qlow=planar_ik(.7,.06,seed=qhigh)
cube=s[:3].copy();base=[cube[0]+.82,cube[1],np.pi]
move([2.1,cube[1],np.pi],qhigh,0,90)
move(base,qhigh,0,45)
move(base,qlow,0,45)
log('contact')
move(base,qlow,1,15)
move(base,qhigh,1,60)
log('lift')
shift=np.array([.95,0.])-s[:2]
base[:2]=np.asarray(base[:2])+shift
move(base,qhigh,1,55)
log('transport')
move(base,qhigh,0,70)
log('release')
env.close()
