from env_client import make_env
import numpy as np
for seed,options in [(i,None) for i in range(5)]+[(42,{'object_count':n}) for n in [1,3,0,50]]:
 e=make_env()
 try:
  s,info=e.reset(seed=seed,options=options)
  print('RESET',seed,options,info,flush=True)
  for name in ['robot','scoop_0','bin_green_0','bin_yellow_0']:
   o=s.get_object_from_name(name)
   fs=['pos_base_x','pos_base_y','pos_base_rot'] if name=='robot' else ['x','y','z','qw','qx','qy','qz']
   print(name,np.round([s.get(o,f) for f in fs],4),flush=True)
 except Exception as ex: print('ERROR',str(ex)[:300],flush=True)
 finally:e.close()
