from env_client import make_env
import numpy as np
env = make_env()
for seed in range(20):
    obs, info = env.reset(seed=seed)
    rob = obs.get_object_from_name('robot')
    a=np.zeros(11,dtype=np.float32)
    obs2,r,term,tr,_=env.step(a)
    cubes=[n for n in obs.get_object_names() if n.startswith('cube')]
    cs=np.array([[obs.get(obs.get_object_from_name(n),'x'),obs.get(obs.get_object_from_name(n),'y')] for n in cubes])
    w=obs.get_object_from_name('wiper_0')
    print(seed, info['object_count'], 'r',r, 'cubes x[%.2f,%.2f] y[%.2f,%.2f]'%(cs[:,0].min(),cs[:,0].max(),cs[:,1].min(),cs[:,1].max()),
      'wiper (%.2f,%.2f,qz%.2f)'%(obs.get(w,'x'),obs.get(w,'y'),obs.get(w,'qz')), 'robot (%.2f,%.2f,%.2f)'%(obs.get(rob,'pos_base_x'),obs.get(rob,'pos_base_y'),obs.get(rob,'pos_base_rot')))
env.close()
