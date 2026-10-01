from env_client import make_env
import numpy as np
for seed,axis,sign in [(0,1,-1),(1,1,-1),(1,0,-1)]:
 e=make_env();s,info=e.reset(seed=seed);ro=s.get_object_from_name('robot');cu=[s.get_object_from_name(n) for n in s.get_object_names() if n.startswith('cube')]
 def coords():return [[o.name]+[round(float(s.get(o,f)),3) for f in ['x','y']] for o in cu]
 x=np.mean([s.get(o,'x') for o in cu]);y=np.mean([s.get(o,'y') for o in cu]);
 targets=[(x,s.get(ro,'pos_base_y')),(x,-.5)] if axis==1 else [(2.2,s.get(ro,'pos_base_y')),(2.2,y),(.3,y)]
 print('INIT',seed,axis,coords(),flush=True)
 for tx,ty in targets:
  for k in range(70):
   d=np.array([tx-s.get(ro,'pos_base_x'),ty-s.get(ro,'pos_base_y')]);a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(d,-.06,.06);s,r,t,tr,i=e.step(a)
   if k%10==0 or t:print('PROGRESS',k,[round(s.get(ro,f),3) for f in ['pos_base_x','pos_base_y']],r,t,coords(),flush=True)
   if max(abs(d))<.015 or t or tr:break
  if t or tr:break
 e.close()
