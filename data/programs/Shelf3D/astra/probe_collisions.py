import numpy as np
from env_client import make_env
from probe_dynamics import vals

def base(s):
 o=s.get_object_from_name('robot');return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
for grip in [0,1]:
 e=make_env();s,_=e.reset(seed=42)
 objects=sorted(n for n in s.get_object_names() if n.startswith('cube'))
 c=s.get_object_from_name(objects[0]);cx=float(s.get(c,'x'));cy=float(s.get(c,'y'))
 print('INIT',grip,vals(s),flush=True)
 for target in [(cx-.7,cy,0),(cx+.7,cy,0),(1.5,0,0),(2.5,0,0)]:
  for t in range(35):
   a=np.zeros(11);a[:3]=np.clip(np.array(target)-base(s),-.1,.1);a[-1]=grip
   s,r,term,trunc,info=e.step(a)
  print('AT',target,'r',r,'state',vals(s),flush=True)
 e.close()
for axis in [3,4,5,6,7,8,9]:
 e=make_env();s,_=e.reset(seed=42)
 for direction in [1,-1]:
  for t in range(125 if direction==1 else 250):
   a=np.zeros(11);a[axis]=direction*.1
   s,r,term,trunc,info=e.step(a)
  print('SAT',axis,direction,vals(s)['robot'],flush=True)
 e.close()
