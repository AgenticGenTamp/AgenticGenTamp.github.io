import numpy as np
from plib2 import *
from env_client import make_env
env=make_env()
obs,info=env.reset(seed=0)
for n in ['part0','part1']:
    o=obs.get_object_from_name(n)
    print(n,o.type.name,{f:round(obs.get(o,f),4) for f in obs.type_features[o.type]})
def run(part,delta,R=Rdown,seed=0):
    obs,info=env.reset(seed=seed); base=rb(obs)
    obs,g,m=try_grasp(env,obs,part,delta,R,base)
    return int(g),m
grid={}
for dy in np.arange(-0.06,0.061,0.02):
    row=[]
    for dx in np.arange(-0.06,0.061,0.02):
        g,m=run('part1',[float(dx),float(dy),0.01])
        row.append(g if m in('ok',) or g else ('B' if 'blocked' in m else 0))
    print("dy%.2f"%dy,row,flush=True)
env.close()
