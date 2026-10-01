import numpy as np
from probe_vis_lib import *
env=make_env()
obs,info=env.reset(seed=0, options={'object_count':1})
print("info",info, [n for n in obs.get_object_names() if n.startswith('objective')])
s=env.get_state()
print("state type",type(s))
try:
    names=sorted(s.get_object_names()); print("state objs",len(names))
    r0=s.get_object_from_name('rover0')
    print("rover0 feats",{f:round(float(s.get(r0,f)),3) for f in s.type_features[r0.type]})
    s.set(r0,'x',0.5); s.set(r0,'y',1.0)
    env.set_state(s)
    obs,_,_,_,_=st(env,op='noop')
    print("after set_state pose",pose(obs,0))
except Exception as e:
    print("ERR",type(e).__name__,e)
env.close()
