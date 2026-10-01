from env_client import make_env
import numpy as np
from scipy.spatial.transform import Rotation as R
E=make_env();s,info=E.reset(seed=0,options={'object_count':1});p=s.get_object_from_name('part0');robot=s.get_object_from_name('robot')
q=[0,.378469836586,-np.pi,-2.05185294,0,-.711269877003,np.pi/2]
dest=np.r_[s.get(p,'pose_x')-.455583437544,s.get(p,'pose_y')+.023653142324,0,q]
for i in range(8):
 cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
 a=np.r_[np.clip(dest-cur,-.2,.2),-1];s,*_=E.step(a)
 if s.get(p,'grasp_active'):break
for delta in [-.06]*5:
 a=np.zeros(11);a[4]=a[8]=delta;s,*_=E.step(a)
print('lifted',s.data[p].tolist(),flush=True)
for i in range(8):
 a=np.zeros(11);a[9]=.2;s,*_=E.step(a)
 print('yaw',i+1,s.get(robot,'joint_7'),'part',s.data[p][:7].tolist(),flush=True)
E.close()
