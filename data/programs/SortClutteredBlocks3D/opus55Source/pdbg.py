import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import approach as A
n=20; sd=int(sys.argv[1]); T=int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False}, design_counts=[n], eval_counts=[n])
s,info=env.reset(seed=sd, options={'object_count': n})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s,info)
for i in range(T):
    s,*_=env.step(ap.get_action(s))
ch=ap._choose_cube(s)
print('movable',ap.movable, 'best score', ap.last_cands[0][4])
for c in ap.movable: print(c, ap._pos_cache[c][:3].round(3), round(np.degrees(A._yaw_from_quat(*ap._pos_cache[c][3:7]))))
print(ap._plan_push(s))
