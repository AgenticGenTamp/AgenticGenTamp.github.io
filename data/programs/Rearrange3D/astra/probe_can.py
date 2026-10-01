from env_client import make_env
from kinematics_candidate import planar_ik,forward
import numpy as np,sys
seed=int(sys.argv[1]);z=float(sys.argv[2]) if len(sys.argv)>2 else .12
s=None;e=make_env();s,_=e.reset(seed=seed);orig=s[32:35].copy()
off=float(sys.argv[4]) if len(sys.argv)>4 else -.10
b=np.array([s[32]-.65+off,s[33]+.3,0]);q0=s[96:103].copy()
def move(q,g,n):
 global s
 for _ in range(n):
  a=np.clip(np.r_[b,q,g]-s[93:104],-.1,.1);a[-1]=g;s,r,t,tr,i=e.step(a)
 return np.round(s[32:35],4).tolist(),np.round(forward(s[96:103],s[93:96],mount_height=0)[0],4).tolist()
qhi=planar_ik(.65,.3,mount_height=0);qlo=planar_ik(.65,z,mount_height=0)
if len(sys.argv)>3:qhi[-1]=qlo[-1]=float(sys.argv[3])
print('START',seed,z,orig,flush=True)
print('aside',move(q0,0,25),flush=True)
print('high',move(qhi,0,100),flush=True)
b[0]=s[32]-.65+off;b[1]=s[33]-.001
print('above',move(qhi,0,30),flush=True)
print('low',move(qlo,0,65),flush=True)
print('close',move(qlo,1,25),flush=True)
print('lift',move(qhi,1,60),flush=True)
b[1]+=.12
print('translate',move(qhi,1,20),flush=True)
e.close()
