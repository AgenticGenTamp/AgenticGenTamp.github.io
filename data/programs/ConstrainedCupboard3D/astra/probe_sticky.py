from env_client import make_env
from approach import GeneratedApproach
import numpy as np
e=make_env();s,info=e.reset(seed=0,options={'object_count':1})
a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info)
for t in range(500):
 s,r,te,tr,info=e.step(a.get_action(s))
 if te:
  print('FIRST',t,a.xyz(s,a.objects[0]),te,flush=True);break
else:
 print('NO_SUCCESS',flush=True);e.close();raise SystemExit
act=np.zeros(11);act[0]=-.08;act[10]=1
for k in range(15):
 s,r,te,tr,info=e.step(act)
 print('PULL',k,a.xyz(s,a.objects[0]),te,tr,flush=True)
e.close()
