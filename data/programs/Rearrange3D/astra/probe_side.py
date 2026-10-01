from env_client import make_env
from kinematics_candidate import planar_ik,forward
import numpy as np,sys
seed=int(sys.argv[1]);idx=int(sys.argv[2]);z=float(sys.argv[3]);q7=float(sys.argv[4]);off=float(sys.argv[5]) if len(sys.argv)>5 else -.12
e=make_env();s,_=e.reset(seed=seed);orig=s[idx:idx+3].copy();b=np.array([s[idx]-.65+off,s[idx+1]+.3,0]);q0=s[96:103].copy()
def move(q,g,n):
 global s
 for _ in range(n):
  a=np.clip(np.r_[b,q,g]-s[93:104],-.1,.1);a[-1]=g;s,r,t,tr,i=e.step(a)
 return np.round(s[idx:idx+7],4).tolist(),np.round(forward(s[96:103],s[93:96],mount_height=0)[0],4).tolist()
qhi=planar_ik(.65,.27,mount_height=0);qlo=planar_ik(.65,z,mount_height=0);qhi[-1]=qlo[-1]=q7
print('START',seed,z,q7,orig,flush=True)
print('aside',move(q0,0,30),flush=True);print('high',move(qhi,0,100),flush=True)
b[0]=s[idx]-.65+off;b[1]=s[idx+1]-.001
print('above',move(qhi,0,30),flush=True);print('low',move(qlo,0,65),flush=True);print('close',move(qlo,1,25),flush=True);print('lift',move(qhi,1,60),flush=True)
b[1]+=.12;print('translate',move(qhi,1,20),flush=True);e.close()
