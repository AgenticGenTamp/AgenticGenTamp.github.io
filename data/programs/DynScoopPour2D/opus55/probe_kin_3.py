import numpy as np
from env_client import make_env
env=make_env()
def R(o): r=o.get_object_from_name('robot'); return {f: round(o.get(r,f),4) for f in ['x','y','theta','arm_joint','finger_gap']}
def objs(o): return {n:np.array([o.get(o.get_object_from_name(n),'x'),o.get(o.get_object_from_name(n),'y')]) for n in o.get_object_names() if n.startswith('small')}
def st(obs,a): return env.step(np.array(a,dtype=float))[0]
obs,_=env.reset(seed=0)
th=[]
for i in range(80):
    obs=st(obs,[0,0,0.098,0,0]); th.append(round(R(obs)['theta'],3))
print('theta seq',th[25:40], th[60:])
# pile interaction: move to x=0.9 at y=1.2 then down
obs,_=env.reset(seed=0)
for i in range(40): obs=st(obs,[-0.03,-0.03,0,0,0])
print('pos',R(obs))
o0=objs(obs)
for i in range(60):
    prev=R(obs); obs=st(obs,[0,-0.03,0,0,0])
    if R(obs)==prev: print('blocked at step',i); break
print('after down',R(obs))
o1=objs(obs)
d={n:np.round(o1[n]-o0[n],3) for n in o0 if np.linalg.norm(o1[n]-o0[n])>1e-3}
print('moved objs',len(d),list(d.items())[:8])
