import numpy as np
from env_client import make_env
from ctrl import move, RDOWN
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
obs,e,u=move(env,obs,[0.500,0.074,0.35],steps=300,grip=0.0)
print("high:",e,u,"q",obs[128:135])
tot=u
for z in np.arange(0.30,-0.05,-0.02):
    obs,e,u=move(env,obs,[0.500,0.074,z],steps=80,grip=0.0); tot+=u
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    print(f"z={z:.2f} err={e:.3f} used={u} cubes={np.round(d,3)} q2={obs[129]:.3f}")
    if d.max()>0.005: break
print("total steps",tot)
env.close()
