from env_client import make_env
from grasp import HookGrasp
import numpy as np
e=make_env();s,i=e.reset(seed=0);p=HookGrasp(e.action_space,e.observation_space);p.reset(s);h=s.get_object_from_name('hook');rb=s.get_object_from_name('robot')
for k in range(160):
 s,r,d,tr,i=e.step(p.get_action(s))
 if s.get(h,'held'):break
for label,a in [('init',[0,0,0,0,0]),('plus',[0,0,0,.099,0]),('plus2',[0,0,0,.099,0]),('plus3',[0,0,0,.099,0]),('minus',[0,0,0,-.099,0])]:
 s,r,d,tr,i=e.step(np.array(a,dtype=np.float32))
 print(label,'robot',*[s.get(rb,f) for f in ['x','y','theta','arm_joint']],'hook',*[s.get(h,f) for f in ['x','y','held']],flush=True)
e.close()
