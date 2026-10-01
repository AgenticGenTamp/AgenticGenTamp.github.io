from env_client import make_env
import numpy as np
for seed in range(10):
 e=make_env();s,i=e.reset(seed=seed)
 def summary(s):
  return {n:{f:round(s.get(s.get_object_from_name(n),f),3) for f in ['x','y','theta']} for n in s.get_object_names()}
 print(seed, 'maxsteps',e.max_steps, summary(s),flush=True)
 t=s.get_object_from_name('target_block')
 for step in range(50):
  s,r,d,tr,i=e.step(np.zeros(5,dtype=np.float32))
  if d or tr:break
 print('idle',step+1,d,tr, {f:round(s.get(t,f),3) for f in ['x','y','theta','vx','vy']},flush=True)
 e.close()
