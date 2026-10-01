from env_client import make_env
import numpy as np

def val(s,o,f): return float(s.get(o,f))

def dump(s, step):
    print('\nSTEP', step)
    for o in sorted(s.data, key=lambda q:q.name):
        typ=o.type.name
        if typ in ('kin_robot','hook','lobject','small_circle','small_square'):
            fs=['x','y','theta']
            if typ == 'kin_robot': fs += ['arm_length','finger_gap']
            else: fs += ['held']
            print(typ, getattr(o,'name',str(o)), {f:round(val(s,o,f),4) for f in fs})

env=make_env(); s,info=env.reset(seed=0); print('max',env.max_steps, 'info', info); print(s.get_object_names()); dump(s,0)
for i in range(3):
    s,r,t,tr,info=env.step(np.zeros(5,dtype=np.float32)); dump(s,i+1)
env.close()
