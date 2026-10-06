import sys, numpy as np, test_approach as T, approach as ap
env=T.make_env()
n,seed,t0,t1=map(int,sys.argv[1:5])
state,info=env.reset(seed=seed,options={'object_count':n})
A=ap.GeneratedApproach(env.action_space,env.observation_space,{}); A.reset(state,info)
for t in range(t1):
    a=A.get_action(state)
    if t>=t0 and A.phase=='carry':
        be,ye=A._base_err(state); c,u,_=A._rod(state,A.cur)
        print(t,f"be={be:.3f} ye={ye:.3f} arm={A._arm_err(state):.3f} u={np.round(u,3)} bt={np.round(A.base_t,3)}")
    state,*_=env.step(a)
