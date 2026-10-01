import numpy as np, sys
from env_client import make_env
import approach as A
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
# wrap _go and _set_grip to log
orig_go=ap._go; orig_grip=ap._set_grip
log=[]
def go(*a,**k):
    n=[0]
    gen=orig_go(*a,**k)
    for x in gen:
        n[0]+=1; yield x
    log.append(("go",tuple(np.round(np.asarray(a[0],float),3)),n[0]))
def grip(*a,**k):
    n=0
    for x in orig_grip(*a,**k):
        n+=1; yield x
    log.append(("grip",a[0],n))
ap._go=go; ap._set_grip=grip
ap.reset(obs,info)
for t in range(1000):
    a=ap.get_action(obs)
    obs,r,te,tr,_=env.step(a)
    if te: print("TERM",t); break
for e in log: print(e)
env.close()
