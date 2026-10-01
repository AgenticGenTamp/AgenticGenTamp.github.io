import json
import numpy as np
from env_client import make_env
from kinova import planar_ik
cases=[(1.6,-.05,.03),(1.6,-.05,.008)]
cases += [(x,z,.008) for z in [-.08,-.05,-.02] for x in [1.55,1.57,1.59,1.61,1.63,1.65]]
for k,(x,z,speed) in enumerate(cases):
 env=make_env();s,_=env.reset(seed=0)
 q=planar_ik(.6,z,pitch=-np.pi/2);q[6]+=np.pi/2
 def move(b,g,n,step=.05):
  global s
  for i in range(n):
   a=np.zeros(11);a[:3]=np.clip(np.asarray(b)-s[125:128],-step,step)
   a[3:10]=np.clip(q-s[128:135],-.1,.1);a[10]=g
   s,r,t,tr,inf=env.step(np.clip(a,env.action_space.low,env.action_space.high))
 move([2.,0.,np.pi],0,150)
 move([x,0.,np.pi],0,45)
 move([x,0.,np.pi],1,15)
 history=[]
 for i in range(60 if speed==.008 else 25):
  move([x+.4,0.,np.pi],1,1,speed)
  history.append(round(float(s[107]),4))
 print(json.dumps(dict(case=k,x=x,z=z,speed=speed,drawer=float(s[107]),maxdrawer=max(history),history=history)),flush=True)
 np.save('drawer_refine_last.npy',s)
 if s[107]>.22:
  np.save('drawer_refine_success.npy',s);env.close();break
 env.close()
