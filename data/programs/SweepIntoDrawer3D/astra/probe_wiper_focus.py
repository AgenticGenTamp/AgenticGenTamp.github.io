import numpy as np
from env_client import make_env
e=make_env()
s,_=e.reset(seed=0); q=s[128:135].copy(); xy=s[147:149]+[.33,0]
for k in range(12):
 a=np.zeros(11,np.float32);a[:2]=np.clip(1.2*(xy-s[125:127]),-.1,.1);a[3:10]=np.clip(2.5*(q-s[128:135]),-.1,.1)
 s,*_=e.step(a)
np.save('wiper_focus_before.npy',s)
for k in range(3):
 a=np.zeros(11,np.float32);a[10]=1;s,*_=e.step(a)
np.save('wiper_focus_grasp.npy',s)
for k in range(5):
 a=np.zeros(11,np.float32);a[0]=.05;a[10]=1;s,*_=e.step(a)
np.save('wiper_focus_pull.npy',s)
print('robot',s[125:136],'wiper',s[147:154],'drawers',s[103:109],flush=True)
e.close()
