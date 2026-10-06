import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
n=int(sys.argv[1]); sd=int(sys.argv[2]); T=int(sys.argv[3])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[n], eval_counts=[n])
ap=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=sd, options={'object_count': n}); ap.reset(s,info)
for i in range(T):
    a=ap.get_action(s)
    s,r,te,tr,_=env.step(a)
    b,arm,g=ap._robot(s)
    ee=ap._ee(b,arm)
    c=ap._pose(s,ap.cur)[:3] if ap.cur else None
    if 1 or (len(sys.argv)>4 and i>=int(sys.argv[4])): print(i, ap.phase, ap.cur, 'ee',ee.round(3), 'cube',None if c is None else c.round(3), 'g',round(g,3), 'tgt', ap.goal_pos.round(3) if hasattr(ap,'goal_pos') else None, 'qerr', np.abs(ap.arm_target-arm).max().round(3))
