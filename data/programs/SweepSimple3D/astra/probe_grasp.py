from env_client import make_env
import numpy as np
E=make_env(); s,i=E.reset(seed=1)
r=s.get_object_from_name('robot'); w=s.get_object_from_name('wiper_0')
def pos(o):return np.array([s.get(o,k) for k in ('x','y','z')])
for t in range(50):
 a=np.zeros(11); a[10]=1 if t>12 else 0
 a[1]=-.04 if t<25 else .04
 s,re,te,tr,i=E.step(a)
 print(t,round(s.get(r,'pos_base_y'),3),np.round(pos(w),3),round(s.get(r,'pos_gripper'),3),re,te,flush=True)
 if te:break
E.close()
