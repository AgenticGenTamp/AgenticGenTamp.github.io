from env_client import make_env
from approach import GeneratedApproach

for seed in range(8):
    env=make_env(); state,info=env.reset(seed=seed,options={'object_count':12})
    app=GeneratedApproach(env.action_space,env.observation_space,{})
    app.reset(state,info)
    for step in range(env.max_steps):
        state,reward,term,trunc,inf=env.step(app.get_action(state))
        if term or trunc: break
    rob=state.get_object_from_name('robot')
    xy=(float(state.get(rob,'pos_base_x')),float(state.get(rob,'pos_base_y')))
    print(seed,step+1,term,trunc,tuple(round(v,2) for v in xy),len(app.path),flush=True)
    env.close()
