from env_client import make_env
from kinova import planar_ik
import numpy as np
senv=make_env();s,_=senv.reset(seed=0)
q=planar_ik(.6,-.05,-np.pi/2);q[6]+=np.pi/2

def move(x,n,g):
 global s
 for k in range(n):
  a=np.zeros(11);a[:3]=np.clip([x,0,np.pi]-s[125:128],-.025,.025);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=g;s,r,d,tr,i=senv.step(a)
move(2.1,160,0)
for j in range(6):
 x=1.6+s[107]
 move(x,40,0);print('before',j,s[103:109],flush=True)
 move(x,12,1);move(x+.4,50,1)
 print('after',j,s[103:109],flush=True)
np.save('regrasp_final.npy',s);senv.close()
