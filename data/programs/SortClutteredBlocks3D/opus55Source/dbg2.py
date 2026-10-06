import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, BIN_ORDER
n=int(sys.argv[1]); sd=int(sys.argv[2]); T=int(sys.argv[3])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SortClutteredBlocks3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[n], eval_counts=[n])
ap=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=sd, options={'object_count': n}); ap.reset(s,info)
b0={b:ap._pose(s,b) for b in BIN_ORDER}
for i in range(T):
    a=ap.get_action(s); ph=ap.phase
    s,r,te,tr,_=env.step(a)
    b,arm,g=ap._robot(s); ee=ap._ee(b,arm)
    for bn in BIN_ORDER:
        p=ap._pose(s,bn)
        if np.abs(p[:3]-b0[bn][:3]).max()>0.004 or abs(abs(p[3])-1)>0.002:
            print(i, 'BIN MOVED', bn, (p[:3]-b0[bn][:3]).round(3), p[3:].round(2), 'phase', ph, 'ee', ee.round(3), 'tgt', ap.goal_pos.round(3), 'cur', ap.cur)
            b0[bn]=p
    if te: break
