import numpy as np
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=4,suppress=True,linewidth=200)
import sys
g=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
env=make_env(); obs,_=env.reset(seed=0)
c0=obs[:80].reshape(5,16)[:,:3].copy()
obs,e=move(env,obs,[0.500,0.074,0.35],steps=60,grip=g)
print("grip",obs[135],"err",e)
for z in np.arange(0.30,0.05,-0.01):
    obs,e=move(env,obs,[0.500,0.074,z],steps=15,grip=g)
    c=obs[:80].reshape(5,16)[:,:3]
    d=np.linalg.norm(c-c0,axis=1)
    print(f"z={z:.2f} err={e:.3f} cubes={np.round(d,3)}")
    if e>0.06 or d.max()>0.01: break
env.close()
