import time, numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
for N in [10,100]:
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:ScoopPour3DEnv",count_kwarg="num_objects",count_object_prefix="cube_",constant_object_env_kwargs={'scene_bg': False},design_counts=[N],eval_counts=[N])
    t=time.time(); s,info=env.reset(seed=0, options={'object_count':N}); print('reset',time.time()-t)
    t=time.time()
    for i in range(50): env.step(np.zeros(11,dtype=np.float32))
    print(N,'step', (time.time()-t)/50)
    print([a for a in dir(env) if not a.startswith('__')][:40])
    try: print('max steps', env._max_steps if hasattr(env,'_max_steps') else None, getattr(env,'max_episode_steps',None))
    except Exception as e: print(e)
