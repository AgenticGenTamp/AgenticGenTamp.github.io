from env_client import make_env
from kin import fk,ik,HOME
from scipy.spatial.transform import Rotation
import numpy as np,sys
mount=float(sys.argv[1]) if len(sys.argv)>1 else .5
length=float(sys.argv[2]) if len(sys.argv)>2 else .15
mode=float(sys.argv[3]) if len(sys.argv)>3 else 1
axis=float(sys.argv[4]) if len(sys.argv)>4 else 0
e=make_env();s,info=e.reset(seed=0);rob=s.get_object_from_name('robot');obj=s.get_object_from_name('cuboid_0')
p=np.array([s.get(obj,f) for f in ['x','y','z']]);quat=[s.get(obj,f) for f in ['qx','qy','qz','qw']];yaw=Rotation.from_quat(quat).as_euler('xyz')[2]
rot=Rotation.from_euler('z',yaw+axis-np.pi/2).as_matrix()@np.diag([1.,-1.,-1.])
qt=np.array([s.get(rob,'pos_arm_joint'+str(j)) for j in range(1,8)])
base=np.array([p[0],p[1]-.55,np.pi/2]);count=0

def move(q,grip,label):
 global s,qt,count
 for j in range(160):
  currbase=np.array([s.get(rob,'pos_base_x'),s.get(rob,'pos_base_y'),s.get(rob,'pos_base_rot')])
  a=np.zeros(11);a[:3]=np.clip((base-currbase)/np.array([.87,.87,1]),-.1,.1);a[3:10]=np.clip(4*(q-qt),-.1,.1);qt+=.25*a[3:10];a[10]=grip
  s,r,te,tr,i=e.step(a);count+=1
  if np.linalg.norm(q-qt)<.001 and np.linalg.norm(base-currbase)<.005:
   for k in range(6):
    a[:10]=0;s,r,te,tr,i=e.step(a);count+=1
   break
 print(label,'steps',count,'reward',r,'obj',[round(s.get(obj,f),4) for f in ['x','y','z']],'robot',[round(s.get(rob,'pos_arm_joint'+str(j)),3) for j in range(1,8)],flush=True)
 print(e.render_state(state=s,label='low_'+label),flush=True)
 return q
q=ik([.4,0,.2-mount],rot,[0,1.7,np.pi,-1.4,0,-.4,0],gripper_length=length)[0];move(q,1-mode,'above')
q=ik([.4,0,-.005-mount],rot,q,gripper_length=length)[0];move(q,1-mode,'lower')
move(q,mode,'close')
q=ik([.4,0,.3-mount],rot,q,gripper_length=length)[0];move(q,mode,'lift')
e.close()
