from env_client import make_env
from fk_scan_backup import GeneratedApproach
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
angles=[math.pi,math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,math.pi/2,-math.pi/2,math.pi]
xyz=np.array([(0,0,.15643),(0,.005375,-.12838),(0,-.21038,-.006375),(0,.006375,-.21038),(0,-.20843,-.006375),(0,.00017505,-.10593),(0,-.10593,-.00017505),(0,0,-.061525)])
def design(q):
 rot=np.eye(3); blocks=[]
 for a,j in zip(angles,list(q)+[0]):
  blocks.append(rot.copy()); rot=rot@R.from_euler('x',a).as_matrix()@R.from_euler('z',j).as_matrix()
 return np.concatenate(blocks,axis=1),rot
E=make_env();s,i=E.reset(seed=0);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i)
for k in range(300):
 s,r,t,tr,i=E.step(p.get_action(s))
 if s.get(p.r,'grasp_active'):break
else:raise RuntimeError('No grasp')
print('GRASP',k,flush=True)
rng=np.random.default_rng(2026);As=[];bs=[];qs=[]
for k in range(150):
 q=np.array([s.get(p.r,'joint_'+str(j)) for j in range(1,8)])
 tf=np.array([s.get(p.r,'grasp_tf_'+f) for f in ['x','y','z']]);rtf=R.from_quat([s.get(p.r,'grasp_tf_'+f) for f in ['qx','qy','qz','qw']])
 rc=R.from_quat([s.get(p.c,'pose_'+f) for f in ['qx','qy','qz','qw']]);re=rc*rtf.inv()
 te=np.array([s.get(p.c,'pose_'+f) for f in ['x','y','z']])-re.apply(tf)
 te[:2]-=[s.get(p.r,'pos_base_x'),s.get(p.r,'pos_base_y')]
 rb=R.from_euler('z',s.get(p.r,'pos_base_rot'));te=rb.inv().apply(te);re=rb.inv()*re
 A,ori=design(q);As.append(A);bs.append(te);qs.append(q)
 if k%20==0:print('SAMPLE',k,'q',q.tolist(),'eef',te.tolist(),'rerr',np.linalg.norm(ori-re.as_matrix()),'active',s.get(p.r,'grasp_active'),flush=True)
 a=np.zeros(11);a[3:10]=rng.uniform(-.4,.4,7);a[10]=-1
 s,r,t,tr,i=E.step(a)
 if not s.get(p.r,'grasp_active'):raise RuntimeError('Lost grasp')
 if t and k<5:print('TERMINATED',k,flush=True)
A=np.concatenate(As);b=np.concatenate(bs);prior=xyz.ravel();x=prior+np.linalg.lstsq(A,b-A@prior,rcond=1e-8)[0]
print('RANK',np.linalg.matrix_rank(A),'RMSE',np.sqrt(np.mean((A@x-b)**2)),'MAXERR',np.max(np.abs(A@x-b)),flush=True)
print('ORIGINS',x.reshape(8,3).tolist(),flush=True)
np.savez('fk_calibration.npz',q=np.array(qs),eef=np.array(bs),origins=x.reshape(8,3),design=np.array(As))
E.close()
