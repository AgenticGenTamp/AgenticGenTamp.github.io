from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=0);rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0')
features=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]
target=np.array([-.22,.287928909,0,0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2])
def pose():return np.array([s.get(p,k) for k in ['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw','grasp_active']])
def act(a):
 global s
 s,r,te,tr,info=E.step(a);return te
for k in range(12):
 a=np.zeros(11);a[:10]=np.clip(target-np.array([s.get(rob,k) for k in features]),-.2,.2);act(a)
a=np.zeros(11);a[10]=-1;act(a);print('START',pose(),[s.get(rob,k) for k in features],flush=True)
for j in [0,1,3,4,5,6,7,8,9]:
 before=pose();a=np.zeros(11);a[j]=.02;act(a);print('D',j,pose()-before,flush=True);a[j]=-.02;act(a)
E.close()
