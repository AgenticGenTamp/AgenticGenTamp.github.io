from env_client import make_env
from kinematics import fk,ik
import numpy as np
E=make_env();s,_=E.reset(seed=0)
def rob(s):
 o=s.get_object_from_name('robot');return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]])
def report(k):
 r=rob(s);print(k,'ee',np.round(fk(r[3:],r[:3])[:3,3],4),'q',np.round(r[3:],3),'objects',[(n,np.round([s.get(s.get_object_from_name(n),f) for f in ['x','y','z']],4).tolist()) for n in ['scoop_0','bin_yellow_0','cube_0']],flush=True)
# Grab scoop with several progressively lower heights.
for h in [.7,.60,.56,.52,.48]:
 r=rob(s); sp=s.get_object_from_name('scoop_0'); p=[s.get(sp,'x'),s.get(sp,'y'),h]
 qt,err=ik(p,np.diag([1.,-1.,-1.]),r[3:],r[:3])
 for k in range(90):
  r=rob(s);d=qt-r[3:];a=np.zeros(11);a[3:10]=d*min(3,.1/max(abs(d)));s,rw,t,tr,i=E.step(a)
 report(h)
 a=np.zeros(11);a[-1]=1
 for k in range(5):s,rw,t,tr,i=E.step(a)
 report('close')
 # lift .07 with close
 r=rob(s);qt,err=ik(np.array(p)+[0,0,.07],np.diag([1.,-1.,-1.]),r[3:],r[:3])
 for k in range(35):
  r=rob(s);d=qt-r[3:];a=np.zeros(11);a[-1]=1;a[3:10]=d*min(3,.1/max(abs(d)));s,rw,t,tr,i=E.step(a)
 report('lift')
E.close()
