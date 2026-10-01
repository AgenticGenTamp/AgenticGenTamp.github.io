from env_client import make_env
from kinematics_candidate import planar_ik,forward
import numpy as np,sys
z=float(sys.argv[1]);g=float(sys.argv[2]);axis=int(sys.argv[3])
e=make_env();s,_=e.reset(seed=0);ob=s[16:19].copy();base=np.array([ob[0]-.65,ob[1]-.001,0])
qhi=planar_ik(.65,.32,mount_height=0);qlo=planar_ik(.65,z,mount_height=0)
def move(b,q,n):
 global s
 for _ in range(n):
  a=np.clip(np.r_[b,q,g]-s[93:104],-.1,.1);a[-1]=g
  s,r,t,tr,i=e.step(a)
move(base,qhi,110)
base[axis]-=.15;move(base,qhi,20);move(base,qlo,60)
for d in np.arange(0,.32,.02):
 b=base.copy();b[axis]+=d;move(b,qlo,8)
 print(round(d-.15,3),np.round(s[16:19]-ob,4).tolist(),'tip',np.round(forward(s[96:103],s[93:96],mount_height=0)[0],3).tolist(),flush=True)
e.close()
