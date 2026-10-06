import sys, math, numpy as np
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from robocode.environments.ss_pybullet import *
from approach import GeneratedApproach
seed, T, r = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=seed)
a = GeneratedApproach(env.action_space, env.observation_space, {})
a.reset(state, info)
b = env.current_backend
orig = b._apply_operator
def patched(index, op):
    if index == r:
        cam = b._camera_point(index)
        conf = get_joint_positions(b._rovers[index], b._base_joints[index])
        print("op", op, "conf", np.round(conf,3), "cam", np.round(cam,3),
              [int(b.image_visible(index, o)) for o in b._objectives],
              [round(float(np.linalg.norm(np.array(get_point(o)) - cam)),3) for o in b._objectives])
    return orig(index, op)
b._apply_operator = patched
for k in range(T):
    print("step", k)
    state, *_ = env.step(a.get_action(state))
