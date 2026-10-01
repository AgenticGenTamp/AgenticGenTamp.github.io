import numpy as np, sys
from env_client import make_env
import approach
from approach import GeneratedApproach
mode = sys.argv[1]
class A(GeneratedApproach):
    def _plan_stage(self, state):
        acts = super()._plan_stage(state)
        if mode=='close' and self.stage=='down' and acts:
            acts[-1]=(acts[-1][0], -1.0)
        if mode=='open' and self.stage=='place' and acts:
            acts[-1]=(acts[-1][0], 1.0)
        return acts
env=make_env()
for seed in [0,1]:
    obs,info=env.reset(seed=seed); ap=A(env.action_space, env.observation_space,{}); ap.reset(obs,info)
    for t in range(100):
        st=ap.stage; a=ap.get_action(obs); obs,r,term,_,_=env.step(a)
        R=obs.get_object_from_name('robot')
        if a[10]!=0: print(seed,t,st,a[10],'ga',obs.get(R,'grasp_active'))
        if term: print('term',t+1); break
