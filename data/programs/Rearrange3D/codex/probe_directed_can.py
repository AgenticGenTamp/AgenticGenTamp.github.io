from env_client import make_env
import numpy as np

env=make_env(); s,_=env.reset(seed=0); idx=np.array([0,1,2,16,17,18,32,33,34]); s0=s.copy()
def st(a,n,label):
 global s
 for k in range(n):
  old=s;s,r,t,tr,_=env.step(np.array(a,np.float32)); dd=s[idx]-old[idx]
  if np.max(np.abs(dd))>.002 or r!=-1:
   print(label,k,'r',r,'base',np.round(s[93:96],3),'obj',np.round(s[idx].reshape(3,3),3),'dd',np.round(dd.reshape(3,3),3),flush=True)
# get behind can at y=-.55 with safe home arm
st([0,-.1,0,0,0,0,0,0,0,0,0],7,'behind')
# configure q2=.68, q4=-1.43
for k in range(70):
 a=np.zeros(11,np.float32)
 if abs(s[97]-.68)>.04:a[4]=.1*np.sign(.68-s[97])
 if abs(s[99]+1.43)>.04:a[6]=.1*np.sign(-1.43-s[99])
 if not np.any(a):break
 st(a,1,'config')
print('configured',np.round(s[[93,94,95,97,99]],3),flush=True)
# push until fingers straddle can, close, then carry/push toward bowl
st([0,.01,0,0,0,0,0,0,0,0,0],34,'push')
st([0,0,0,0,0,0,0,0,0,0,1],10,'close')
st([0,.01,0,0,0,0,0,0,0,0,1],40,'carry')
st([0,0,0,0,0,0,0,0,0,0,0],5,'settle')
env.close()
