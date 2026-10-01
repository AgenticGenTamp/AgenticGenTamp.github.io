import sys, numpy as np
from env_client import make_env
from probe_term_lib import *
seed=int(sys.argv[1]); theta=np.radians(float(sys.argv[2])); push=float(sys.argv[3]) if len(sys.argv)>3 else 0.08
R=1.0
env=make_env(); obs,info=env.reset(seed=seed)
cn=chairs(obs)[0]; c0=cxy(obs,cn); r0=rxy(obs)
print('reset chair',{k:round(v,4) for k,v in cfull(obs,cn).items()})
print('reset robot',rxy(obs))
sa=np.arctan2(*(r0-c0)[::-1])
obs,term,_=move_to(env,obs,c0+R*np.array([np.cos(sa),np.sin(sa)]))
obs,term=orbit_to(env,obs,c0,theta+np.pi,R)
print('after position: chair',{k:round(v,4) for k,v in cfull(obs,cn).items()},'robot',rxy(obs))
d=np.array([np.cos(theta),np.sin(theta)])
for i in range(150):
    obs,rew,term,trunc,info=env.step(act(d[0]*push,d[1]*push))
    cf=cfull(obs,cn); r=rxy(obs)
    dist=np.linalg.norm(np.array([cf['x'],cf['y']])-r)
    tilt=np.degrees(2*np.arccos(min(1,abs(cf['qw']))))
    print(f"{i:3d} rew={rew} term={term} c=({cf['x']:.3f},{cf['y']:.3f},{cf['z']:.3f}) tilt={tilt:5.1f} v=({cf['vx']:.2f},{cf['vy']:.2f},{cf['vz']:.2f}) w=({cf['wx']:.2f},{cf['wy']:.2f},{cf['wz']:.2f}) rob=({r[0]:.3f},{r[1]:.3f}) dist={dist:.3f}",flush=True)
    if term or trunc: break
env.close()
