from env_client import make_env
import numpy as np
E=make_env()
counts={}
for seed in range(50):
 s,info=E.reset(seed=seed); n=info['object_count']; counts[n]=counts.get(n,0)+1
 if seed<2:
  for i in range(3):
   s,r,t,tr,info=E.step(np.zeros(11))
   print('noop',seed,i,r,t,tr,[(n,float(s.get(s.get_object_from_name(n),'pose_z'))) for n in s.get_object_names() if n!='robot'])
print('counts',counts)
for n in [0,1,3,5,10]:
 try:
  s,info=E.reset(seed=0,options={'object_count':n})
  print('forced',n,info,[(name,round(s.get(s.get_object_from_name(name),'pose_x'),3),round(s.get(s.get_object_from_name(name),'pose_y'),3)) for name in s.get_object_names() if name.startswith('part')])
 except Exception as exc: print(type(exc),str(exc))
E.close()
