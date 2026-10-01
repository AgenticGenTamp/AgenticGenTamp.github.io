import json
import numpy as np
from env_client import make_env
from kinova import planar_ik, fk

env=make_env();s,_=env.reset(seed=0)
def move(b,q,g,n):
 global s
 for i in range(n):
  a=np.zeros(11);a[:3]=np.clip(np.asarray(b)-s[125:128],-.05,.05)
  a[3:10]=np.clip(q-s[128:135],-.1,.1);a[10]=g
  s,r,t,tr,inf=env.step(np.clip(a,env.action_space.low,env.action_space.high))

def trial(x,y,z,roll,first=False):
 q=planar_ik(.6,z,pitch=-np.pi/2);q[6]+=roll
 move([2.,y,np.pi],q,0,150 if first else 35)
 move([x,y,np.pi],q,0,40)
 np.save('drawer_y_contact.npy',s)
 move([x,y,np.pi],q,1,12)
 move([x+.3,y,np.pi],q,1,35)
 result=dict(x=x,y=y,z=z,roll=roll,drawer=s[103:109].round(5).tolist(),qerror=round(float(np.max(np.abs(q-s[128:135]))),4))
 print(json.dumps(result),flush=True)
 np.save('drawer_y_last.npy',s)
 if np.max(s[103:109])>.015:
  np.save('drawer_y_success.npy',s)
  env.close();raise SystemExit

first=True
for roll in [np.pi/2,-np.pi/2]:
 for z in [-.05,-.1,0.]:
  for x in [1.8,1.7,1.6,1.85]:
   trial(x,0.,z,roll,first);first=False
for z in [-.05,-.1,0.]:
 for x in [1.8,1.55]:
  for y in [-.25,-.15,-.1,-.05,.05,.1,.15,.25]:
   trial(x,y,z,0.)
env.close()
