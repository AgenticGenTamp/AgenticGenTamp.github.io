import numpy as np, sys
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0)
res={}
for z in [0.30,0.20,0.10,0.0,-0.10,-0.20]:
    obs,e=move(env,obs,[0.20,0.0,z],steps=60,grip=1.0)
    if e>0.05:
        print(f"z={z} cannot retract, err={e}"); continue
    stall=None
    for x in np.arange(0.22,0.72,0.02):
        obs,e=move(env,obs,[x,0.0,z],steps=12,grip=1.0)
        if e>0.05:
            stall=x; break
    print(f"z_local={z:+.2f} stall_x={stall} err={e:.3f}")
    obs,e=move(env,obs,[0.20,0.0,z],steps=40,grip=1.0)
env.close()
