import sys, test_approach as T, approach as ap
env=T.make_env()
n,seed=int(sys.argv[1]),int(sys.argv[2])
state,info=env.reset(seed=seed,options={'object_count':n})
A=ap.GeneratedApproach(env.action_space,env.observation_space,{}); A.reset(state,info)
for t in range(int(sys.argv[3])):
    a=A.get_action(state)
    if A.phase in ('approach','descend'):
        be,ye=A._base_err(state); print(t,A.phase,f"be={be:.3f} ye={ye:.3f} arm={A._arm_err(state):.3f}")
    state,*_=env.step(a)
