from env_client import make_env
import numpy as np

e=make_env(); fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
for k in range(10):
 if k in [2,7,9]:continue
 for sign in [-1,1]:
  o,_=e.reset(seed=0);r=o.get_object_from_name('robot')
  # Put base in empty centre to avoid table interference.
  for _ in range(13):
   a=np.zeros(11);a[0]=np.clip(-o.get(r,'base_x'),-.2,.2);o,*_=e.step(a)
  if k==0:
   for _ in range(10):
    a=np.zeros(11);a[1]=.2;o,*_=e.step(a)
  old=float(o.get(r,fs[k]));n=0
  for t in range(80):
   a=np.zeros(11);a[k]=sign*.2;o,*_=e.step(a);v=float(o.get(r,fs[k]))
   if abs(v-old)<1e-6:n+=1
   else:n=0
   if n>=2:break
   old=v
  print(fs[k],sign,v,'steps',t+1,flush=True)
e.close()
