import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq
env=make_env()
obs,info=env.reset(seed=42)
obs,n,ok=grasp_seq(env,obs)
print("first grasp",ok)
# release
for _ in range(15): obs,_,_,_,_=env.step(act(dg=0.02))
print("released, held=",oget(obs,'hook','held'),"hook",round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3))
# re-close without moving
for k in range(12):
    obs,_,_,_,_=env.step(act(dg=-0.02))
    if oget(obs,'hook','held')>0.5: print("re-grasp OK at gap",round(rget(obs,'finger_gap'),3)); break
else: print("re-grasp FAILED")
env.close()
