import numpy as np,sys
from env_client import make_env
from ctrl import *
env=make_env(); obs,info=env.reset(seed=0)
o=np.asarray(obs);print("start",np.round(o[93:96],3))
# phase1: rotate to 0
obs=base_goto(env,obs,o[93],o[94],0.0,steps=15)
print("rot",np.round(np.asarray(obs)[93:96],3))
# phase2: y
obs=base_goto(env,obs,np.asarray(obs)[93],-0.323,0.0,steps=20)
print("y",np.round(np.asarray(obs)[93:96],3))
# phase3: x slow
for tx in [-0.18,-0.14,-0.10,-0.06]:
    obs=base_goto(env,obs,tx,-0.323,0.0,steps=15,vmax=0.04)
    print("x->",tx,np.round(np.asarray(obs)[93:96],3))
env.close()
