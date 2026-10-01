"""Test whether the discovered contact is reachable without its random prefix."""
import numpy as np
from env_client import make_env
FS=tuple("pos_arm_joint%d"%i for i in range(1,8))
Q=np.array([-.33330977,1.59132075,.27980691,1.32690668,.38126963,-.09951454,1.67638755])
A=np.array([-.00013796837,-.000042617321,-.00062416802,-.1,.1,-.1,-.1,.1,.1,.1,1.,-12.,10.235282,-10.266092,-.81716883,4.211204,12.,2.9062495],np.float32)
def v(s,n,fs):
 o=s.get_object_from_name(n); return np.array([float(s.get(o,f)) for f in fs])
acts=np.load("random_contact_found.npz")["actions"]
Q0=np.array([.9441,-.0616,1.4739,1.3419,-.5038,-1.5938,.2053])
for alpha,scale in ((-.3,.8),(-.6,.8),(-.9,.8),(-1.2,.8),(-1.5,.8),(-1.8,.8)):
 grip_setup=1.
 env=make_env();s,_=env.reset(seed=0,options={"object_count":1});p0=v(s,"cube_0",("x","y","z")); goal=p0[:2]-[.6,.25]
 ca,sa=np.cos(alpha),np.sin(alpha); rot=np.array([[ca,-sa],[sa,ca]]); goal=p0[:2]-rot@np.array([.6,.25])
 for t in range(70):
  q=v(s,"robot",FS);b=v(s,"robot",("pos_base_x","pos_base_y"));a=np.zeros(18,np.float32)
  a[:2]=np.clip(goal-b,-.05,.05);a[2]=np.clip(.4799+alpha-v(s,"robot",("pos_base_rot",))[0],-.05,.05);a[3:10]=np.clip(Q0-q,-.05,.05);a[10]=grip_setup;a[11:]=np.clip(4*(Q0-q),-4,4);s,*_=env.step(a)
 pre=v(s,"cube_0",("x","y","z")); qb=v(s,"robot",FS)
 for a0 in acts[139:145]:
  a=a0.copy();a[:2]=rot@a[:2];a[10]=1.;a[11:18]*=scale;s,*_=env.step(a)
 post=v(s,"cube_0",("x","y","z"))
 impact=post-pre; vel=v(s,"cube_0",("vx","vy","vz")); maxz=post[2]
 for _ in range(12):
  z=np.zeros(18,np.float32);z[10]=1.;s,*_=env.step(z);maxz=max(maxz,v(s,"cube_0",("z",))[0])
 final=v(s,"cube_0",("x","y","z"))-pre
 print("alpha",alpha,"scale",scale,"impact",np.round(impact,4),"vel",np.round(vel,3),"final",np.round(final,4),"maxz",round(maxz,4),flush=True);env.close()
