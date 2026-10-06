import numpy as np, logging, time
logging.disable(logging.WARNING)
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from kinder.envs.kinematic3d.transport3d import ObjectCentricTransport3DEnv
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube",
    constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
s, info = env.reset(seed=3, options={'object_count': 3})
print(type(s), info)
t=time.time()
sim = ObjectCentricTransport3DEnv(num_cubes=3, num_boxes=1, allow_state_access=True)
sim.reset(seed=0)
print("sim create", time.time()-t)
sim.set_state(s)
a = np.zeros(11, dtype=np.float32); a[0]=0.1; a[3]=0.1
s1,*_ = env.step(a); s2,*_ = sim.step(a)
print(type(s2), np.allclose(s1.data if hasattr(s1,'data') else 0, 0))
for n in ['robot','cube0']:
    o1=s1.get_object_from_name(n); o2=s2.get_object_from_name(n)
    print(n, s1[o1] if hasattr(s1,'__getitem__') else None, s2[o2])
print(env._max_episode_steps if hasattr(env,'_max_episode_steps') else None, [k for k in vars(env).keys()])
