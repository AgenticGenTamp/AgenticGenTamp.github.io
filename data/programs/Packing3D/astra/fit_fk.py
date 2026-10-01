import json,numpy as np
from scipy.spatial.transform import Rotation as R
exec(open('probe_fk.py').read().split('t=np.eye')[0])
data=json.load(open('calibration_data.json'));A=[];b=[]
for d in data:
 rob=d['robot'];obj=d['part'];q=[rob['joint_'+str(i)] for i in range(1,8)]
 t=np.eye(4)
 for (xyz,rx),qi in zip(params,q):
  a=np.eye(4);a[:3,3]=xyz;a[:3,:3]=R.from_euler('x',rx).as_matrix()@R.from_euler('z',qi).as_matrix();t=t@a
 rot=R.from_quat([obj['pose_q'+c] for c in 'xyzw']).as_matrix()@R.from_quat([rob['grasp_tf_q'+c] for c in 'xyzw']).as_matrix().T
 ee=np.array([obj['pose_'+c] for c in 'xyz'])-rot@np.array([rob['grasp_tf_'+c] for c in 'xyz'])
 rz=R.from_euler('z',rob['pos_base_rot']).as_matrix()
 target=rz.T@(ee-np.array([rob['pos_base_x'],rob['pos_base_y'],0]))-t[:3,3]
 A.append(np.c_[np.eye(3),t[:3,:3]]);b.append(target)
x=np.linalg.lstsq(np.vstack(A),np.hstack(b),rcond=None)[0]
print('base offset and tool offset',x,'maxerr',np.max(np.abs(np.vstack(A)@x-np.hstack(b))))
