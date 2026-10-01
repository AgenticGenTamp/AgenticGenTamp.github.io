from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=42)
r=s.get_objects(E.observation_space.get_type('kin_robot'))[0]
def report(tag):
 print(tag, {f:round(s.get(r,f),3) for f in ('x','y','theta','base_radius','arm_joint','arm_length','finger_gap','finger_height')},flush=True)
def act(a,n):
 global s
 for _ in range(n):s,_,_,_,_=E.step(np.array(a,dtype=float))
 report((a,n))
report('start')
act([0,.03,0,0,0],20)
act([0,0,.098,0,0],18)
act([0,0,0,.08,0],20)
act([0,0,0,-.08,0],20)
act([-.03,0,0,0,0],80)
act([0,-.03,0,0,0],90)
act([.03,0,0,0,0],100)
print('small',[(round(s.get(o,'x'),3),round(s.get(o,'y'),3)) for t in ('small_circle','small_square') for o in s.get_objects(E.observation_space.get_type(t))])
E.close()
