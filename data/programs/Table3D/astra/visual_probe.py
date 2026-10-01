from env_client import make_env
import numpy as np
from scipy.spatial.transform import Rotation
import math
xyz=[(0,0,.15643),(0,.005375,-.12838),(0,-.21038,-.006375),(0,.006375,-.21038),(0,-.20843,-.006375),(0,.00017505,-.10593),(0,-.10593,-.00017505)]
angles=[math.pi,math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,math.pi/2,-math.pi/2]
def fk(q):
 t=np.eye(4)
 for p,a,j in zip(xyz,angles,q):
  n=np.eye(4);n[:3,3]=p;n[:3,:3]=Rotation.from_euler('x',a).as_matrix()@Rotation.from_euler('z',j).as_matrix();t=t@n
 n=np.eye(4);n[:3,3]=(0,0,-.061525);n[:3,:3]=Rotation.from_euler('x',math.pi).as_matrix()
 return t@n
q=np.array([0,-.35,-math.pi,-2.5,0,-.87,math.pi/2]);print('fk home',fk(q),flush=True)
e=make_env();s,_=e.reset(seed=0,options={'object_count':1});r=s.get_objects(e.observation_space.get_type('Kinematic3DRobot'))[0];cube=s.get_object_from_name('cube0');target=np.array([s.get(cube,'pose_x'),s.get(cube,'pose_y')]);fields=['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]
for q4 in np.linspace(-2.55,-.8,15):
 s,_=e.reset(seed=0,options={'object_count':1});desired=q.copy();desired[3]=q4
 f=fk(desired);print('q4',q4,'fk',f[:3,3],flush=True)
 # Base scan along x, y minor correction, fingers 10cm extension estimated
 pos=f[:3,3]+f[:3,:3]@np.array([0,0,.12]);base=target-pos[:2]
 for dx in np.linspace(-.18,.18,13):
  for dy in [-.06,0,.06]:
   goal=np.r_[base+[dx,dy],0,desired]
   for k in range(20):
    cur=np.array([s.get(r,v) for v in fields]);delta=goal-cur
    if max(abs(delta))<.0001:break
    a=np.r_[np.clip(delta,-.4,.4),1];s,_,t,tr,_=e.step(a)
   a=np.zeros(11);a[-1]=-1;s,_,t,tr,_=e.step(a)
   if s.get(r,'grasp_active'):
    print('GRASP',q4,dx,dy,{v:s.get(r,v) for v in fields},'tf',[s.get(r,'grasp_tf_'+v) for v in ['x','y','z','qx','qy','qz','qw']],flush=True);e.close();raise SystemExit
print('NO GRASP',flush=True);e.close()
