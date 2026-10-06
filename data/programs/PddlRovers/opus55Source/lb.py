import sys, approach
approach.R_TALL = float(sys.argv[1])
if len(sys.argv) > 4:
    approach.MARGIN = 0.002; approach.REL_MIN = 1.6; approach.RNG_MARGIN = 0.0
    _orig = approach.GeneratedApproach.reset

from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
tot = 0; out = []
for s in range(int(sys.argv[2]), int(sys.argv[3])):
    st, info = env.reset(seed=s)
    a = approach.GeneratedApproach(None, None, {})
    a.reset(st, info)
    out.append(a.cost2d)
print(sys.argv[1], sum(out)/len(out), out)
