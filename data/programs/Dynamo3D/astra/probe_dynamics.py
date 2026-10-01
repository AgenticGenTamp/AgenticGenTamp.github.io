import numpy as np
from env_client import make_env

env=make_env()
s, info=env.reset(seed=0)
t=env.observation_space.get_type('mujoco_tidybot_robot')
r=next(iter(s.get_objects(t)))
features=['pos_base_x','pos_base_y','pos_base_rot','vel_base_x','vel_base_y','vel_base_rot']
def pos(s):return [round(s.get(r,f),5) for f in features]
print('INFO',info,'MAX',env.max_steps,flush=True)
for name in sorted(s.get_object_names()):
 o=s.get_object_from_name(name)
 print('OBJ',name,[(f,round(s.get(o,f),4)) for f in env.observation_space.type_features[o.type]],flush=True)
print('START',pos(s),flush=True)
for axis, value, count in [(0,.1,5),(0,0,3),(0,-.1,5),(1,.1,5),(1,-.1,5),(2,.1,10),(0,.1,5),(1,.1,5)]:
 for i in range(count):
  a=np.zeros(11,dtype=np.float32);a[axis]=value
  s,re,te,tr,info=env.step(a)
  print('STEP',axis,value,i,pos(s),'R',re,'D',te,tr,info,flush=True)
  if te or tr:break
env.close()
