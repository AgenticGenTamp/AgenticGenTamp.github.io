import numpy as np
from env_client import make_env
np.set_printoptions(precision=3,suppress=True)
e=make_env()
def move(s,xy,g=0,n=12):
 for k in range(n):
  a=np.zeros(11,np.float32); a[:2]=np.clip((np.array(xy)-s[125:127])*1.2,-.1,.1);a[10]=g
  s,r,t,tr,i=e.step(a)
 return s
for dx in np.arange(.2,.701,.1):
 for dy in np.arange(-.2,.201,.1):
  s,_=e.reset(seed=0); w=s[147:150].copy(); xy=w[:2]+[dx,dy]
  s=move(s,xy); before=s[147:150].copy()
  a=np.zeros(11,np.float32); a[10]=1
  for k in range(3):s,*_=e.step(a)
  after=s[147:150].copy();s=move(s,xy+[.18,0],g=1,n=5)
  diff=s[147:150]-after
  print('offset',np.round([dx,dy],2),'pre',np.round(before-w,3),'grasp',np.round(after-before,3),'pull',np.round(diff,3),'drawers',np.round(s[103:109],3),flush=True)
  if np.linalg.norm(diff)>.06:
   np.save('wiper_candidate_'+str(round(dx,1))+'_'+str(round(dy,1))+'.npy',s)
e.close()
