import numpy as np
from scipy.spatial.transform import Rotation
from env_client import make_env
from kinematics import ik,fk,DOWN,Q0

e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name('block1');f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
def v():return np.array([s.get(r,k) for k in f])
def xyz(o):return np.array([s.get(o,'pose_'+k) for k in 'xyz'])
def move(t,grip=0):
 global s
 for _ in range(40):
  prev=v();s,*rest=e.step(np.r_[np.clip(t-prev,-.1,.1),grip])
  if np.linalg.norm(v()-t)<1e-4:return True
  if np.linalg.norm(v()-prev)<1e-6:return False
 return False
def q(z,yaw=0):return ik(np.array([.72,.25,z]),h=.990675,q0=v()[3:],rot=Rotation.from_euler('z',yaw).as_matrix()@DOWN)
p=xyz(b);base=np.r_[p[:2]-[.67,.25],0];yaw=2*np.arctan2(s.get(b,'pose_qz'),s.get(b,'pose_qw'))
for z in [.97,.88,.835]:print('reach',z,move(np.r_[base,q(z,yaw)],1),flush=True)
s,*_=e.step(np.r_[np.zeros(10),-1]);print('grasp',s.get(r,'grasp_active'),flush=True)
for z in [.87,.93,1.0]:
 print('lift',z,move(np.r_[base,q(z,yaw)]),flush=True)
 rel,rot=fk(v()[3:],h=.990675);tool=rel+np.r_[v()[:2],0]-[.05,0,0];tf=np.array([s.get(r,'grasp_tf_'+k) for k in 'xyz']);print('fk error',xyz(b)-(tool+rot@tf),flush=True)
print('rotate',move(np.r_[base,q(1.0)]),flush=True)
for loc in [[-.08,-.08],[0,0],[.08,.08]]:
 p=xyz(b);desired=np.r_[loc,p[2]];newbase=v()[:3].copy();newbase[:2]+=desired[:2]-p[:2]
 print('move above',loc,move(np.r_[newbase,v()[3:]]),flush=True)
print('opening high at',xyz(b),flush=True);s,*_=e.step(np.r_[np.zeros(10),1]);print('drop active',s.get(r,'grasp_active'),'block',[s.get(b,'pose_'+k) for k in ['x','y','z','qx','qy','qz','qw']],flush=True)
e.close()
