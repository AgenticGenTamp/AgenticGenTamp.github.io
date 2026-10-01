"""Repeat the directed distilled collision while tracking the moving cube."""
import numpy as np
from env_client import make_env
FS=tuple("pos_arm_joint%d"%i for i in range(1,8)); Q=np.array([.9441,-.0616,1.4739,1.3419,-.5038,-1.5938,.2053])
def v(s,n,fs): o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
alpha=-1.305; ca,sa=np.cos(alpha),np.sin(alpha);R=np.array([[ca,-sa],[sa,ca]]); off=R@np.array([.6,.25]); acts=np.load("random_contact_found.npz")["actions"][139:145]
env=make_env();s,_=env.reset(seed=0,options={"object_count":1});pinit=v(s,"cube_0",("x","y","z")); binp=v(s,"bin_0",("x","y","z"))
for cycle in range(16):
 for t in range(55 if cycle==0 else 32):
  p=v(s,"cube_0",("x","y"));b=v(s,"robot",("pos_base_x","pos_base_y"));q=v(s,"robot",FS);a=np.zeros(18,np.float32)
  a[:2]=np.clip(p-off-b,-.06,.06);a[2]=np.clip(.4799+alpha-v(s,"robot",("pos_base_rot",))[0],-.06,.06);a[3:10]=np.clip(Q-q,-.06,.06);a[10]=1.;a[11:]=np.clip(4*(Q-q),-4,4);s,r,term,trunc,_=env.step(a)
 for a0 in acts:
  a=a0.copy();a[:2]=R@a[:2];a[10]=1.;s,r,term,trunc,_=env.step(a)
 for _ in range(8):
  a=np.zeros(18,np.float32);a[10]=1.;s,r,term,trunc,_=env.step(a)
 p=v(s,"cube_0",("x","y","z"));print(cycle,np.round(p,3),"dinit",round(float(np.linalg.norm(p[:2]-pinit[:2])),3),"dbin",round(float(np.linalg.norm(p[:2]-binp[:2])),3),"r",r,flush=True)
 if term or trunc: break
env.close()
