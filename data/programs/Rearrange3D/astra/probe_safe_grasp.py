from env_client import make_env
from kinematics_candidate import planar_ik,forward
import numpy as np,sys
z=float(sys.argv[1]);close=float(sys.argv[2]);q7=float(sys.argv[3]) if len(sys.argv)>3 else 0
s=None;e=make_env();s,_=e.reset(seed=0);orig=s[16:19].copy()
b=np.array([s[16]-.65,s[17]+.3,0]);q0=s[96:103].copy()
def move(q,g,n,b):
 global s
 for _ in range(n):
  a=np.clip(np.r_[b,q,g]-s[93:104],-.1,.1);a[-1]=g;s,r,t,tr,i=e.step(a)
 return np.round(s[16:19],4).tolist(),np.round(forward(s[96:103],s[93:96],mount_height=0)[0],4).tolist()
qhi=planar_ik(.65,.3,mount_height=0);qhi[-1]=q7
qlo=planar_ik(.65,z,mount_height=0);qlo[-1]=q7
print('START',z,close,q7,flush=True)
print('aside',move(q0,1-close,25,b),flush=True)
print('high',move(qhi,1-close,100,b),flush=True)
b[0]=s[16]-.65;b[1]=s[17]-.001
print('above',move(qhi,1-close,30,b),flush=True)
print('low',move(qlo,1-close,65,b),flush=True)
print('close',move(qlo,close,25,b),flush=True)
print('lift',move(qhi,close,60,b),flush=True)
print('translate',move(qhi,close,20,b+np.array([0,-.12,0])),flush=True)
e.close()
