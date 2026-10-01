from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
e=make_env(); o,_=e.reset(seed=0)
print('init',o,'limit',e.max_steps)
for name,a,n in [('extend',[0,0,0,.1,0],12),('north',[0,.05,0,0,0],20),('east',[.05,0,0,0,0],20),('south',[0,-.05,0,0,0],20),('retract',[0,0,0,-.1,0],12)]:
 for k in range(n):
  p=o.copy();o,r,t,tr,i=e.step(a)
  if k in (0,n-1):print(name,k,'robot',o[:9], 'objects',o[[9,10,11,20,21,29,30]],'reward',r,t,tr)
e.close()
