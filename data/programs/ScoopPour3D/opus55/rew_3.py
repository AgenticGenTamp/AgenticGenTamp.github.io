import numpy as np
from calib_util import *
from env_client import make_env
for N in [10,20]:
  r=R.__new__(R); r.env=make_env(); r.obs,_=r.env.reset(seed=0,options={'object_count':N})
  o=r.obs.get_object_from_name('scoop_0'); print('scoop',[round(r.obs.get(o,f),3) for f in ['x','y','z','qw','qx','qy','qz','bb_x','bb_y','bb_z']])
  o=r.obs.get_object_from_name('bin_green_0'); print('green',[round(r.obs.get(o,f),3) for f in ['x','y','z','bb_x','bb_y','bb_z']])
  print(sorted((n,tuple(r.P(n).round(3))) for n in r.obs.get_object_names() if n.startswith('cube')))
