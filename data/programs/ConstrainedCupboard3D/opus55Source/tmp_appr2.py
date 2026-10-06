import sys, numpy as np, test_approach as T, approach as ap
env=T.make_env()
n,seed,t0,t1=map(int,sys.argv[1:5])
state,info=env.reset(seed=seed,options={'object_count':n})
A=ap.GeneratedApproach(env.action_space,env.observation_space,{}); A.reset(state,info)
for t in range(t1):
    a=A.get_action(state)
    if t>=t0:
        b,_=A._read(state); print(t,A.phase,A.cur,'base',np.round(b,2),'bt',np.round(A.base_t,2),'staged',getattr(A,'staged',None),'plan',None if A.plan is None else np.round(A.plan[0],2))
    state,*_=env.step(a)
