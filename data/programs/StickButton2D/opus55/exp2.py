from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=1)
r=obs.get_object_from_name('robot'); b=obs.get_object_from_name('button1')
def rs(obs): return [round(float(obs.get(r,f)),3) for f in ['x','y','theta','arm_joint']]
def bs(obs): return [round(float(obs.get(b,f)),3) for f in ['color_r','color_g','color_b']]
for i in range(40):
    x,y=obs.get(r,'x'),obs.get(r,'y')
    tx,ty=obs.get(b,'x'),obs.get(b,'y')
    d=np.array([tx-x,ty-y]); d=np.clip(d,-0.05,0.05)
    obs,rew,term,trunc,_=env.step(np.array([d[0],d[1],0,0,0],dtype=np.float32))
    print(i,rs(obs),bs(obs),rew,term, round(np.hypot(tx-obs.get(r,'x'),ty-obs.get(r,'y')),3))
