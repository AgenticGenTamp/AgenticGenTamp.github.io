from env_client import make_env
from kinova import fk,ik
from scipy.spatial.transform import Rotation as R
import numpy as np
np.set_printoptions(precision=4,suppress=True)
e=make_env();s,_=e.reset(seed=0);q=s[128:135].copy();target=ik([.4,0,.2],R.from_euler('y',-np.pi/2).as_matrix()@fk(q)[:3,:3],q)
print('target',target,flush=True)
for k in range(150):
 a=np.zeros(11);a[0]=min(max(1.8-s[125],-.05),.05);a[3:10]=np.clip(target-s[128:135],-.1,.1);s,*_=e.step(a)
 if k%25==24: print(k,'q',s[128:135],'err',target-s[128:135],flush=True)
np.save('horizontal_final.npy',s);e.close()
