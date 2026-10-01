"""Randomly tune the isolated six-action strike for vertical clearance."""
import numpy as np
from env_client import make_env
FS=tuple("pos_arm_joint%d"%i for i in range(1,8)); Q=np.array([.9441,-.0616,1.4739,1.3419,-.5038,-1.5938,.2053])
def v(s,n,fs): o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
rng=np.random.RandomState(90441); baseacts=np.load("random_contact_found.npz")["actions"][139:145]
env=make_env(); best=(-1,None)
for trial in range(30):
 s,_=env.reset(seed=0,options={"object_count":1});p0=v(s,"cube_0",("x","y","z"));goal=p0[:2]-[.6,.25]
 for _ in range(58):
  q=v(s,"robot",FS);b=v(s,"robot",("pos_base_x","pos_base_y"));a=np.zeros(18,np.float32);a[:2]=np.clip(goal-b,-.06,.06);a[2]=np.clip(.4799-v(s,"robot",("pos_base_rot",))[0],-.06,.06);a[3:10]=np.clip(Q-q,-.06,.06);a[10]=1.;a[11:]=np.clip(4*(Q-q),-4,4);s,*_=env.step(a)
 # One common time scaling plus per-joint gains preserves swing shape.
 gains=np.clip(rng.normal(.82,.16,7),.35,1.25)
 # Occasionally emphasize wrist/elbow joints suspected of upward flick.
 if trial>=15: gains[3:7]*=rng.uniform(.75,1.35,4)
 maxz=p0[2]; maxvz=0.; moved=0.
 used=[]
 for a0 in baseacts:
  a=a0.copy();a[10]=1.;a[11:18]*=gains;used.append(a.copy());s,*_=env.step(a)
  p=v(s,"cube_0",("x","y","z"));maxz=max(maxz,p[2]);maxvz=max(maxvz,v(s,"cube_0",("vz",))[0]);moved=max(moved,np.linalg.norm(p-p0))
 for _ in range(16):
  a=np.zeros(18,np.float32);a[10]=1.;s,*_=env.step(a);maxz=max(maxz,v(s,"cube_0",("z",))[0]);maxvz=max(maxvz,v(s,"cube_0",("vz",))[0])
 score=maxz+max(0,maxvz)*.03
 print(trial,"z",round(maxz,4),"vz",round(maxvz,3),"move",round(float(moved),3),"g",np.round(gains,3),flush=True)
 if moved>.002 and score>best[0]: best=(score,(trial,maxz,maxvz,gains,np.array(used)))
trial,maxz,maxvz,gains,used=best[1];np.savez("vertical_best.npz",actions=used,gains=gains)
print("BEST",trial,"z",maxz,"vz",maxvz,"gains",gains,"actions",repr(used.tolist()),flush=True);env.close()
