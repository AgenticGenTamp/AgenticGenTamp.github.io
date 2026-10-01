from env_client import make_env
import numpy as np
for oc in [10,20,50,100]:
    env = make_env()
    obs, info = env.reset(seed=1, options={'object_count': oc})
    a = np.zeros(11, dtype=np.float32)
    obs, rew, term, trunc, info = env.step(a)
    cubes=[o for o in obs.data if o.name.startswith('cube_')]
    print('oc',oc,'info',info,'ncubes',len(cubes),'rew',rew)
    print('  green', np.round(obs.data[obs.get_object_from_name('bin_green_0')][:3],3),
          'yellow', np.round(obs.data[obs.get_object_from_name('bin_yellow_0')][:3],3),
          'scoop', np.round(obs.data[obs.get_object_from_name('scoop_0')][:3],3))
    env.close()
