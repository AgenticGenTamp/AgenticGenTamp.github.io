import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
res=[]
for seed in [int(x) for x in sys.argv[1:]]:
    env=make_env(); o,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(o,info)
    term=False
    for t in range(env.max_steps):
        o,r,term,tr,_=env.step(ap.get_action(o))
        if term or tr: break
    cubes=sorted([n for n in o.get_object_names() if n.startswith('cube')])
    pos={n:np.round([float(o.get(o.get_object_from_name(n),k)) for k in 'xyz'],2).tolist() for n in cubes}
    print(f"seed={seed} nobj={info['object_count']} steps={t+1} TERM={term} {pos}",flush=True)
    env.close()
