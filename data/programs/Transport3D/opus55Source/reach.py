import logging, numpy as np, math
logging.disable(logging.WARNING)
import approach
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
s, info = env.reset(seed=0, options={'object_count': 1})
sim = approach.Sim(1); sim.load(s)
b = sim.base()
for z in (0.055, 0.43, 0.6):
  for d in (0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
    for rel in (0, math.pi/2):
        j = sim.rel_ik(np.array([0.0, 0.0, 0.0]), np.array([d, 0.0, z]), rel)
        print(f"z={z} d={d} relyaw={rel:.2f} ok={j is not None}")
