import numpy as np
from env_client import make_env
E=make_env()
s,i=E.reset(seed=0)
print('INFO',i,'MAX',E.max_steps)
for n in sorted(s.get_object_names()):
 o=s.get_object_from_name(n)
 print(n, {f:round(s.get(o,f),4) for f in E.observation_space.type_features[o.type]})
for j in range(5):
 s,r,t,u,i=E.step(np.array([.049,.049,0,-.099,.019]))
 o=s.get_objects(E.observation_space.get_type('kin_robot'))[0]
 print(j,{f:round(s.get(o,f),4) for f in ['x','y','theta','arm_joint','arm_length','finger_gap']},r,t,u,i)
E.close()
