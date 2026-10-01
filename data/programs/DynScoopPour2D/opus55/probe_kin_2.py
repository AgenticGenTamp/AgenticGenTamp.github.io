import numpy as np
from env_client import make_env
env=make_env()
def R(o): r=o.get_object_from_name('robot'); return {f: round(o.get(r,f),4) for f in ['x','y','theta','arm_joint','finger_gap']}
def objs(o): return {n:(round(o.get(o.get_object_from_name(n),'x'),3),round(o.get(o.get_object_from_name(n),'y'),3)) for n in o.get_object_names() if n.startswith('small')}
def run(a,n,obs,verbose=False):
    last=None
    for i in range(n):
        obs,r,te,tr,info=env.step(np.array(a,dtype=float))
        cur=R(obs)
        if verbose: print(i,cur)
        if last is not None and cur==last: break
        last=cur
    return obs,i
obs,_=env.reset(seed=0)
# theta wrap in free space
obs,i=run([0,0,0.098,0,0],70,obs); print('spin',i,R(obs))
obs,_=env.reset(seed=0)
obs,i=run([0,0,0,0.08,0],100,obs); print('arm max',i,R(obs))
obs,i=run([0,0,0,-0.08,0],100,obs); print('arm min',i,R(obs))
obs,i=run([0,0,0,0,0.015],100,obs); print('grip max',i,R(obs))
obs,i=run([0,0,0,0,-0.015],100,obs); print('grip min',i,R(obs))
# wall: base at y=1.0 move left from right side
obs,_=env.reset(seed=0)
obs,i=run([0,0,0.098,0,0],32,obs); print('rot',R(obs))  # rotate so arm points ~up?
