from env_client import make_env
from kinematics_candidate import planar_ik,forward
import numpy as np,sys
z=float(sys.argv[1]) if len(sys.argv)>1 else .12
close=float(sys.argv[2]) if len(sys.argv)>2 else 1
xoff=float(sys.argv[3]) if len(sys.argv)>3 else 0
q7=float(sys.argv[4]) if len(sys.argv)>4 else np.pi/2
e=make_env();s,_=e.reset(seed=0);original=s[16:19].copy();base=np.array([s[16]-.65+xoff,s[17]-.001,0])
steps=0

def move(q,g,n,basegoal=base):
 global s,steps
 for _ in range(n):
  a=np.clip(np.r_[basegoal,q,g]-s[93:104],-.1,.1);a[-1]=g
  s,r,t,tr,i=e.step(a);steps+=1
 return np.round(s[16:19],4).tolist(),np.round(forward(s[96:103],s[93:96],mount_height=0)[0],4).tolist()
qhi=planar_ik(.65,.3,mount_height=0);qhi[-1]=q7
qlo=planar_ik(.65,z,mount_height=0);qlo[-1]=q7
print('START',z,close,xoff,q7,original,flush=True)
print('high',move(qhi,1-close,100),flush=True)
print('low',move(qlo,1-close,50),flush=True)
print('close',move(qlo,close,20),flush=True)
print('lift',move(qhi,close,60),flush=True)
print('translate',move(qhi,close,20,base+np.array([0,-.12,0])),flush=True)
e.close()
