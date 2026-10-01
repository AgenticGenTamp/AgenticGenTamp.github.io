import sys
from env_client import make_env
from approach import GeneratedApproach

env=make_env(); state,info=env.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0)
p=GeneratedApproach(env.action_space,env.observation_space,{})
p.reset(state,info)
last_stage = -1
names=[n for n in state.get_object_names() if n.startswith('small_')]
g=lambda n,f: float(state.get(state.get_object_from_name(n),f))
for t in range(1000):
    state,r,done,trunc,info=env.step(p.get_action(state))
    if t%50==49 or done or (p.stage != last_stage and p.stage >= 8):
        print(t+1,'stage',p.stage,'age',p.stage_age,'cycles',p.cycles,
              'best75',p.best_right,'held',g('hook','held'),'robot',
              round(g('robot','x'),2),round(g('robot','y'),2),
              'hook',round(g('hook','x'),2),round(g('hook','y'),2),
              round(g('hook','theta'),2),
              'right2',sum(g(n,'x')>2 for n in names),flush=True)
        last_stage = p.stage
    if done or trunc: break
env.close()
