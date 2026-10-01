from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=0); rob=s.get_object_from_name('robot')
for bx in [-.12,-.17,-.22,-.27,-.07,-.02,.03,.08]:
 for by in np.arange(-.45,.451,.05):
  for _ in range(6):
   a=np.zeros(11);a[0]=np.clip(bx-s.get(rob,'pos_base_x'),-.2,.2);a[1]=np.clip(by-s.get(rob,'pos_base_y'),-.2,.2)
   s,*_=E.step(a)
  a=np.zeros(11);a[10]=-1;s,r,te,tr,info=E.step(a)
  if s.get(rob,'grasp_active'):
   print('GRASP',bx,by,flush=True)
   for t in E.observation_space.types:
    for o in s.get_objects(t):print(o.name,{f:s.get(o,f) for f in E.observation_space.type_features[t]},flush=True)
   E.close();quit()
 print('bx',bx,'actual',s.get(rob,'pos_base_x'),flush=True)
print('NONE');E.close()
