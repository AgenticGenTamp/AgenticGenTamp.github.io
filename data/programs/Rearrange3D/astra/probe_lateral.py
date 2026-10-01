from env_client import make_env
from kinematics_candidate import planar_ik,forward
import numpy as np,sys
z=float(sys.argv[1]);xoff=float(sys.argv[2]);q7=float(sys.argv[3]) if len(sys.argv)>3 else 0
e=make_env();s,_=e.reset(seed=0);orig=s[16:19].copy();b=np.array([s[16]-.65+xoff,s[17]+.25,0]);q0=s[96:103].copy()
def move(q,g,n):
 global s
 for _ in range(n):
  a=np.clip(np.r_[b,q,g]-s[93:104],-.1,.1);a[-1]=g;s,r,t,tr,i=e.step(a)
 return np.round(s[16:19],4).tolist(),np.round(forward(s[96:103],s[93:96],mount_height=0)[0],4).tolist()
qhi=planar_ik(.65,.3,mount_height=0);qhi[-1]=q7
qlo=planar_ik(.65,z,mount_height=0);qlo[-1]=q7
print('start',z,xoff,q7,flush=True)
move(q0,0,30);move(qhi,0,100);move(qlo,0,65)
for yo in np.arange(.25,-.251,-.025):
 b[1]=orig[1]+yo
 print(round(yo,3),move(qlo,0,10),flush=True)
e.close()
