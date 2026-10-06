import sys, numpy as np, test_approach as T, approach as ap
env=T.make_env()
n,seed,T1=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); w=sys.argv[4]
state,info=env.reset(seed=seed,options={'object_count':n})
A=ap.GeneratedApproach(env.action_space,env.observation_space,{}); A.reset(state,info)
last=None; lp=None
for t in range(T1):
    a=A.get_action(state)
    c,u,_=A._rod(state,w)
    if lp is not None and np.linalg.norm(c-lp)>0.003:
        b,_=A._read(state); print(t,A.phase,A.cur,'MOVED',np.round(c,3),np.round(u,2),'base',np.round(b,2))
    lp=c
    state,*_=env.step(a)
