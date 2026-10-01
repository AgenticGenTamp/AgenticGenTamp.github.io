from env_client import make_env
import numpy as np
env = make_env()
for seed in range(20):
    obs, info = env.reset(seed=seed)
    sh = obs.get_object_from_name('shelf')
    x1,w1,y1,h1 = [obs.get(sh,f) for f in ['x1','width1','y1','height1']]
    blocks = obs.get_objects([t for t in env.observation_space.types if t.name=='target_block'][0]) if False else [obs.get_object_from_name(n) for n in obs.get_object_names() if n.startswith('block')]
    ins = sum(1 for b in blocks if obs.get(b,'y')>2.6)
    r = obs.get_object_from_name('robot')
    print(seed, info['object_count'], len(blocks), 'inshelf',ins, 'shelf x1 %.3f w1 %.3f'%(x1,w1), 'robot %.2f %.2f'%(obs.get(r,'x'),obs.get(r,'y')))
env.close()
