from env_client import make_env
from approach import GeneratedApproach
from scipy.spatial.transform import Rotation as R
import numpy as np
E=make_env();s,i=E.reset(seed=0);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i)
for k in range(200):
 s,r,t,tr,i=E.step(p.get_action(s))
 if s.get(p.r,'grasp_active'):
  for o in [p.r,p.c]:
   print(o.name,{f:float(s.get(o,f)) for f in E.observation_space.type_features[o.type]})
  tf=np.array([s.get(p.r,'grasp_tf_'+f) for f in ['x','y','z']]);rot=R.from_quat([s.get(p.r,'grasp_tf_'+f) for f in ['qx','qy','qz','qw']]).inv()
  print('EEF',np.array([s.get(p.c,'pose_'+f) for f in ['x','y','z']])-rot.apply(tf),'ROT',rot.as_matrix())
  break
E.close()
