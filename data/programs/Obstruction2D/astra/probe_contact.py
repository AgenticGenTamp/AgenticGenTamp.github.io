from env_client import make_env
import numpy as np
for arm in [.1,.2]:
 e=make_env();s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');b=s.get_object_from_name('target_block');x=s.get(b,'x')+s.get(b,'width')/2
 for i in range(6):
  s,*_=e.step(np.array([np.clip(x-s.get(rob,'x'),-.05,.05),0,0,arm-s.get(rob,'arm_joint'),0]))
 print('ARM',arm,'target',[(f,s.get(b,f)) for f in ['x','y','width','height']])
 for i in range(70):
  y=s.get(rob,'y');old=s.get(b,'y');s,r,t,tr,_=e.step(np.array([0,-.01,0,0,1]));new=s.get(rob,'y')
  if i%5==0 or abs(new-y)<.005:print(i,'robot',new,'block',s.get(b,'y'),'dy',new-y)
  if abs(new-y)<.005:
   for j in range(3):
    s,*_=e.step(np.array([0,.01,0,0,1]));print('UP',s.get(rob,'y'),s.get(b,'y'))
   break
 e.close()
