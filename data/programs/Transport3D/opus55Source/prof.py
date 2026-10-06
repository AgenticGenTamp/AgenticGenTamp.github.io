import sys, logging, numpy as np
logging.disable(logging.WARNING)
import approach as A
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1], eval_counts=[1])
k=int(sys.argv[1])
ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
for seed in range(int(sys.argv[2])):
    s,info = env.reset(seed=seed, options={'object_count':k})
    ap.reset(s, info)
    # segment string: B=base only, J=joint only, C=combined, G=gripper
    out=""
    for a in ap.plan:
        b = np.abs(a[:3]).max()>1e-7; j=np.abs(a[3:10]).max()>1e-7
        out += "G" if abs(a[10])>0.5 else ("C" if b and j else ("B" if b else ("J" if j else "0")))
    import itertools
    print(seed, len(ap.plan), " ".join(f"{c}{len(list(g))}" for c,g in itertools.groupby(out)))
    print("   ", [l for l in ap.sims[k].log][:6])
