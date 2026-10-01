from env_client import make_env
from kin import fk,ik
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def trial(params):
 mount,axis,z=params; length=.15; mode=1
 e=make_env();s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');obj=s.get_object_from_name('cuboid_0')
 p=np.array([s.get(obj,f) for f in ['x','y','z']]);quat=[s.get(obj,f) for f in ['qx','qy','qz','qw']];yaw=Rotation.from_quat(quat).as_euler('xyz')[2]
 rot=Rotation.from_euler('z',yaw+axis).as_matrix()@np.diag([1.,-1.,-1.])
 qt=np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
 base=np.array([p[0]-.55,p[1],0]); count=0
 def move(q,grip,label):
  nonlocal s,qt,count
  for j in range(160):
   currbase=np.array([s.get(rob,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
   a=np.zeros(11);a[:3]=np.clip((base-currbase)/np.array([.87,.87,1]),-.1,.1);a[3:10]=np.clip(4*(q-qt),-.1,.1);qt+=.25*a[3:10];a[10]=grip
   s,r,te,tr,_=e.step(a);count+=1
   if np.linalg.norm(q-qt)<.001 and np.linalg.norm(base-currbase)<.005:
    for k in range(6): a[:10]=0;s,r,te,tr,_=e.step(a);count+=1
    break
  obsq=np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
  actual=fk(obsq,length)[0]+[base[0]+.15,base[1],mount]
  print('GRID',mount,round(axis,3),z,label,'steps',count,'obj',np.round([s.get(obj,f) for f in ['x','y','z']],4).tolist(),'FK',np.round(actual,4).tolist(),'qerror',round(np.linalg.norm(obsq-q),3),flush=True)
 q=ik([.4,0,.2-mount],rot,[0,1.7,np.pi,-1.4,0,-.4,0],gripper_length=length)[0];move(q,0,'above')
 q=ik([.4,0,z-mount],rot,q,gripper_length=length)[0];move(q,0,'lower')
 move(q,1,'close')
 q=ik([.4,0,.3-mount],rot,q,gripper_length=length)[0];move(q,1,'lift')
 e.close()
if __name__=='__main__':
 params=[(.4,0,.025),(.4,np.pi/2,.025),(.4,0,.005),(.4,np.pi/2,.005)]
 with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(trial,params))
