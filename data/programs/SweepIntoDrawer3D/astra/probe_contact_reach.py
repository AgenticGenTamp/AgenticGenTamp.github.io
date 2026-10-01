import numpy as np
from env_client import make_env
e=make_env();s,_=e.reset(seed=0);q=s[128:135].copy();q[3]+=.3
for k in range(50):
 a=np.zeros(11,np.float32);a[:2]=np.clip(np.array([.65,-.3849])-s[125:127],-.05,.05);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=1;s,*_=e.step(a)
print('near',s[125:136],'wiper',s[147:150],'drawer',s[103:109]);np.save('contact_near.npy',s);e.close()
