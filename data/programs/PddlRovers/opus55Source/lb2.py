import sys, approach, refine3d
approach.RES = float(sys.argv[1]); approach.K = int(round(0.2/approach.RES)); approach.WIN = 2*approach.K+1; refine3d.KX = approach.K
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
out = []
import time
for s in range(int(sys.argv[2]), int(sys.argv[3])):
    st, info = env.reset(seed=s)
    a = approach.GeneratedApproach(None, None, {})
    t=time.time(); a.reset(st, info)
    out.append(a.cost2d); print(s, a.cost2d, a.planned_cost, round(time.time()-t,1), flush=True)
print(sys.argv[1], sum(out)/len(out), out)
