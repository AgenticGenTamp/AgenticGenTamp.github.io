from env_client import make_env
import numpy as np
env = make_env()
rows=[]
for seed in range(40):
    obs, info = env.reset(seed=seed)
    g=lambda n,f: obs.get(obs.get_object_from_name(n),f)
    obst=[o for o in obs.data if o.name.startswith('obstruction')]
    sx,sw=g('target_surface','x'),g('target_surface','width')
    onsurf=sum(1 for o in obst if abs(obs.get(o,'x')-sx)<(sw+obs.get(o,'width'))/2)
    print(seed, len(obst), 'blk w%.3f h%.3f x%.3f'%(g('target_block','width'),g('target_block','height'),g('target_block','x')),
      'surf x%.3f w%.3f'%(sx,sw), 'obs', [(round(obs.get(o,'x'),2),round(obs.get(o,'width'),2),round(obs.get(o,'height'),2)) for o in obst], 'onsurf',onsurf,
      'rob %.2f %.2f %.2f'%(g('robot','x'),g('robot','y'),g('robot','theta')))
env.close()
