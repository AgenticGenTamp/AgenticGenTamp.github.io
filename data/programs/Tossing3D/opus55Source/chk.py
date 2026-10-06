from common import make_env, obj
env=make_env([1,2])
s,_=env.reset(seed=0, options={'object_count':1}); s=s.copy()
o=s.get_object_from_name('cube_0'); s.set(o,'x',1.0); s.set(o,'y',-0.6)
s2,_=env.reset(seed=0, options={'init_state':s}); print(obj(s2,'cube_0')[:3])
env.set_state(s); print(obj(env.get_state(),'cube_0')[:3])
import numpy as np
s3,*_=env.step(np.zeros(18)); print(obj(s3,'cube_0')[:3])
