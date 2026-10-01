from env_client import make_env
from kinematics import ik,fk
import numpy as np
from scipy.spatial.transform import Rotation

def vals(s,o,fs):return np.array([s.get(o,f) for f in fs])
e=make_env();s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');bn=s.get_object_from_name('bin_yellow_0');green=s.get_object_from_name('bin_green_0');cubes=[o for o in s.get_objects(e.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
qf=['pos_arm_joint'+str(i) for i in range(1,8)];vf=['vel_arm_joint'+str(i) for i in range(1,8)];base=vals(s,rob,['pos_base_x','pos_base_y','pos_base_rot']);bp=vals(s,bn,['x','y','z']);rim=bp+[-.22,0,0];R=np.diag([1.,-1.,-1.])
phases=[(rim+[0,0,.20],0,75,R),(rim+[0,0,.032],0,50,R),(rim+[0,0,.032],1,20,R),(rim+[0,0,.15],1,45,R),(rim+[0,.40,.15],1,25,R),(rim+[0,.40,.24],1,60,R),(rim+[0,.40,.24],1,70,Rotation.from_euler('y',70,degrees=True).as_matrix()@R),(rim+[0,.40,.13],0,30,R)]
base_target=base[1]+.4
for phase,(target,g,n,rot) in enumerate(phases):
 if phase==2:target=vals(s,bn,['x','y','z'])+[-.22,0,.032]
 if phase!=4:q,err=ik(target,rot,vals(s,rob,qf),base)
 for j in range(n):
  a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip(2.5*(q-vals(s,rob,qf))+.15*vals(s,rob,vf),-.1,.1);a[10]=g
  if phase==4:a[1]=np.clip((base_target-vals(s,rob,['pos_base_y'])[0])*.8,-.04,.04)
  s,r,t,tr,_=e.step(a)
  base=vals(s,rob,['pos_base_x','pos_base_y','pos_base_rot'])
  if j%20==0 or j==n-1:
   cp=np.array([vals(s,o,['x','y','z']) for o in cubes]);gp=vals(s,green,['x','y','z']);near=np.sum((abs(cp[:,0]-gp[0])<.22)&(abs(cp[:,1]-gp[1])<.15)&(abs(cp[:,2]-gp[2])<.1))
   print(phase,j,'ee',fk(vals(s,rob,qf),base)[:3,3].round(3),'bin',vals(s,bn,['x','y','z']).round(3),'binquat',vals(s,bn,['qw','qx','qy','qz']).round(2),'cube',cp.mean(0).round(3),'near',near,'r',r,'term',t,flush=True)
  if t or tr:break
 if t or tr:break
e.close()
