import numpy as np, sys
from env_client import make_env
from ctrl import move
np.set_printoptions(precision=4,suppress=True,linewidth=200)
# vertical contact test at empty counter spot, gripper closed
env=make_env(); obs,_=env.reset(seed=0)
obs,e=move(env,obs,[0.45,-0.25,0.35],steps=60,grip=1.0)
print("grip",obs[135])
for z in np.arange(0.30,0.05,-0.01):
    obs,e=move(env,obs,[0.45,-0.25,z],steps=15,grip=1.0)
    print(f"z={z:.2f} err={e:.3f}")
    if e>0.05: break
env.close()
