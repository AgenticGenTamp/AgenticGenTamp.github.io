import numpy as np
from env_client import make_env

def run(seed, stages):
 e=make_env();s,i=e.reset(seed=seed)
 rb=s.get_objects(e.observation_space.get_type('kin_robot'))[0]
 tb=s.get_objects(e.observation_space.get_type('target_block'))[0]
 def show(step):
  print(seed,step,'robot',{f:round(s.get(rb,f),3) for f in ['x','y','theta','arm_joint','arm_length','finger_gap']},'target',{f:round(s.get(tb,f),3) for f in ['x','y','theta','held']},flush=True)
 show('init')
 for label,count,a in stages:
  for k in range(count):
   s,r,d,tr,i=e.step(np.array(a,dtype=np.float32))
   if d or tr:print('END',d,tr);break
  show(label)
  if d or tr:break
 e.close()
run(0,[('retract',20,[0,0,0,-.099,0]),('up',60,[0,.049,0,0,0]),('right',100,[.049,0,0,0,0]),('extend',100,[0,0,0,.099,0]),('open',100,[0,0,0,0,.019]),('close',100,[0,0,0,0,-.019]),('down',100,[0,-.049,0,0,0])])
run(0,[('right',70,[.049,0,0,0,0]),('upfar',80,[0,.049,0,0,0]),('turnup',25,[0,0,.064,0,0]),('upturned',70,[0,.049,0,0,0]),('leftabove',80,[-.049,0,0,0,0])])
