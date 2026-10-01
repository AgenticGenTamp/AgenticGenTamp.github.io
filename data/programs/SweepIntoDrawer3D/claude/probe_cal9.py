import numpy as np, sys
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=3,suppress=True,linewidth=200)
# x-sweep at cube3 y, find contact x ; also probe stall z at empty spot
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
obs,e,u=move(env,obs,[0.30,0.074,0.30],steps=300,grip=0.0)
obs,e,u=move(env,obs,[0.30,0.074,0.205],steps=80,grip=0.0)
prev=obs[:80].reshape(5,16)[:,:3].copy()
for x in np.arange(0.32,0.60,0.02):
    obs,e,u=move(env,obs,[x,0.074,0.205],steps=40,grip=0.0)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-prev,axis=1)
    print(f"x={x:.2f} err={e:.3f} moved={np.round(d,3)} cube3={np.round(c[3],3)}")
    prev=c.copy()
env.close()
