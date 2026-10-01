import numpy as np
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
obs,e,u=move(env,obs,[0.500,0.074,0.35],steps=250)
print("reach high:",e,u)
for z in np.arange(0.30,0.10,-0.01):
    obs,e,u=move(env,obs,[0.500,0.074,z],steps=60)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    print(f"z={z:.2f} err={e:.3f} used={u} cubes={np.round(d,3)}")
    if d.max()>0.005: break
env.close()
