import json
import numpy as np
from env_client import make_env
from kinova import planar_ik, fk

for offset in [.7,.6,.8]:
 for closing in [1.,0.]:
  for z in [.08,.04,.12,0.,-.04]:
   env=make_env();s,_=env.reset(seed=0)
   initial=s[:80].reshape(5,16)[:,:3].copy()
   target=initial[0].copy();base=np.array([target[0]+offset,target[1],np.pi])
   def move(b,q,g,n):
    global s
    for i in range(n):
     a=np.zeros(11);a[:3]=np.clip(np.asarray(b)-s[125:128],-.05,.05)
     a[3:10]=np.clip(q-s[128:135],-.10,.10);a[10]=g
     s,r,t,tr,inf=env.step(np.clip(a,env.action_space.low,env.action_space.high))
   high=planar_ik(.5,.2)
   low=planar_ik(.5,z,seed=high)
   move([1.8,target[1],np.pi],high,1-closing,100)
   move(base,high,1-closing,45)
   move(base,low,1-closing,35)
   before=s[:80].reshape(5,16)[:,:3].copy()
   move(base,low,closing,15)
   move(base,high,closing,45)
   final=s[:80].reshape(5,16)[:,:3]
   delta=final-initial
   print(json.dumps(dict(offset=offset,closing=closing,z=z,cube_z=final[:,2].round(3).tolist(),delta=delta.round(3).tolist(),gripper=round(float(s[135]),3),base=s[125:128].round(3).tolist(),fk=fk(s[128:135])[:3,3].round(3).tolist())),flush=True)
   np.save('cube_pick_last.npy',s)
   env.close()
   if np.max(final[:,2])>.55:
    np.save('cube_pick_success.npy',s)
    raise SystemExit
