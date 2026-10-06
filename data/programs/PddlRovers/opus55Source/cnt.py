from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from approach import GeneratedApproach
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
a = GeneratedApproach(None, None, {})
import collections
c = collections.Counter()
for s in range(40):
    st, info = env.reset(seed=s)
    rv, l, o, sm, ob = a._parse(st)
    c[sum(1 for x in ob if x["half_x"] < 0.15)] += 1
print(c)
