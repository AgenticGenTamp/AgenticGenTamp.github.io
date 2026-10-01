from env_client import make_env
import numpy as np
env=make_env(); T=lambda t: env.observation_space.get_type(t)
obs,_=env.reset(seed=0)
r1=[o for o in obs.get_objects(T('rover')) if o.name=='rover1'][0]
S=-5/6
def st(a):
    global obs
    obs,r,te,tr,i=env.step(np.array(a,dtype=np.float32))
    return [round(obs.get(r1,f),3) for f in ['x','y','theta','store_full']]
print(st([0,0,0,0,.1645,-.145,0,0]))
print(st([0,0,0,0,.1645,-.145,0,0]))
print(st([0,0,0,0,.2,0,0,S]))
# test drop + move
print(st([0,0,0,0,-.2,0,0,5/6]))
# move away out of range with sample op: pre in range, post out
print(st([0,0,0,0,.2,0,0,0]))  # at -0.271 now dist 0
print(st([0,0,0,0,-.2,0,0,0]))
print(st([0,0,0,0,-.2,0,0,S]))  # pre dist .2, post .4
