from env_client import make_env
import numpy as np

env=make_env(); ix=np.array([0,1,2,16,17,18,32,33,34])
configs=[(-1.3,-2.55),(-.8,-2.55),(.2,-2.55),(.7,-2.55),(-.8,-1.4),(.2,-1.4),(.7,-1.4)]
for q2,q4 in configs:
 s0,_=env.reset(seed=0); s=s0
 # servo q2(action index4) and q4(index6) approximately with bang-bang feedback
 for k in range(70):
  a=np.zeros(11,np.float32)
  if abs(s[97]-q2)>.04: a[4]=.1*np.sign(q2-s[97])
  if abs(s[99]-q4)>.04: a[6]=.1*np.sign(q4-s[99])
  if not np.any(a): break
  old=s;s,r,t,tr,_=env.step(a)
 # translate to low y then scan across entire lateral interval
 for sign,n in [(-1,6),(1,14)]:
  a=np.zeros(11,np.float32);a[1]=sign*.1
  for k in range(n):
   old=s;s,r,t,tr,_=env.step(a); dd=s[ix]-old[ix]
   if np.max(np.abs(dd))>.003 or r!=-1:
    print('CONTACT cfg',q2,q4,'base',np.round(s[93:96],2),'q',np.round(s[96:103],2),'r',r,'dd',np.round(dd.reshape(3,3),3),flush=True)
 print('done',q2,q4,'actual',np.round(s[[94,97,99]],2),flush=True)
env.close()
