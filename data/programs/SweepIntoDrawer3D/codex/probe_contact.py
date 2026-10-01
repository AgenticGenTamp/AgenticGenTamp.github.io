import numpy as np
from env_client import make_env

e=make_env(); o,_=e.reset(seed=0); home=o[128:135].copy(); wp=o[147:150].copy()
def act_base(tx,ty):
 global o
 for k in range(60):
  a=np.zeros(11,np.float32); a[:2]=np.clip(([tx,ty]-o[125:127])*2,-.1,.1); a[10]=1
  o,*_=e.step(a)
  if np.linalg.norm(o[125:127]-[tx,ty])<.03: break
 print('base',o[125:127])
act_base(o[125],-.75); act_base(1.24,-.75); act_base(1.24,-.38)
rng=np.random.default_rng(4)
for trial in range(30):
 target=home+rng.uniform([-1,-.8,-.5,-.8,-.7,-.7,-1],[1,.8,.5,.8,.7,.7,1])
 for k in range(35):
  a=np.zeros(11,np.float32); a[3:10]=np.clip((target-o[128:135])*.5,-.1,.1); a[10]=1
  o,r,t,tr,inf=e.step(a)
 d=np.linalg.norm(o[147:150]-wp)
 print(trial,'q',np.round(o[128:135],2),'w',np.round(o[147:150],3),'d',round(d,4))
 if d>.01:
  print('CONTACT close now')
  qa=np.array([-.51,.38,2.89,-2.49,.46,-.25,1.99]); qb=np.array([.37,-.44,3.38,-2.58,.6,-.49,1.35])
  for cyc in range(12):
   for dest in (qa,qb):
    for z in range(18):
     a=np.zeros(11,np.float32);a[3:10]=np.clip((dest-o[128:135])*.5,-.1,.1);a[10]=1;o,*_=e.step(a)
   print('cycle',cyc,'w',np.round(o[147:150],3))
  break
e.close()
