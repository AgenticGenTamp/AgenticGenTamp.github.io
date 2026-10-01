from env_client import make_env
from kinematics import ik,fk
import numpy as np

def vals(s,o,fs): return np.array([s.get(o,f) for f in fs])
for h in [.015,-.005]:
 e=make_env();s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');bn=s.get_object_from_name('bin_yellow_0')
 qf=['pos_arm_joint'+str(i) for i in range(1,8)];vf=['vel_arm_joint'+str(i) for i in range(1,8)];base=vals(s,rob,['pos_base_x','pos_base_y','pos_base_rot'])
 bp=vals(s,bn,['x','y','z']);rim=bp+[-.22,0,0]
 for phase,(target,g,n) in enumerate([(rim+[0,0,.20],0,75),(rim+[0,0,h],0,50),(rim+[0,0,h],1,20),(rim+[0,0,.24],1,60)]):
  if phase==2:target=vals(s,bn,['x','y','z'])+[-.22,0,h]
  q,err=ik(target,np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]),vals(s,rob,qf),base)
  for j in range(n):
   a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip(2.5*(q-vals(s,rob,qf))+.15*vals(s,rob,vf),-.1,.1);a[10]=g
   s,r,t,tr,_=e.step(a)
  print('h',h,'phase',phase,'ee',fk(vals(s,rob,qf),base)[:3,3].round(4),'bin',vals(s,bn,['x','y','z']).round(4),'reward',r,flush=True)
 e.close()
