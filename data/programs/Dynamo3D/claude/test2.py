import sys, numpy as np
from env_client import make_env
sys.path.insert(0,'.')
from approach import GeneratedApproach, _nearest_in_rect, RECT_A, RECT_B
lo,hi=int(sys.argv[1]),int(sys.argv[2])
for s in range(lo,hi):
    env=make_env(); obs,info=env.reset(seed=s)
    r=obs.get_object_from_name('robot')
    b0=np.array([float(obs.get(r,'pos_base_x')),float(obs.get(r,'pos_base_y'))])
    n=info.get('object_count')
    rect=RECT_A if n==1 else RECT_B
    t=_nearest_in_rect(rect,b0)
    ideal=int(np.ceil(float(np.max(np.abs(t-b0)))/0.087))
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for i in range(env.max_steps):
        a=ap.get_action(obs); obs,rw,term,trunc,info=env.step(a)
        if term or trunc: break
    print("seed",s,"n",n,"steps",i+1,"ideal",ideal,"over",i+1-ideal,flush=True)
    env.close()
