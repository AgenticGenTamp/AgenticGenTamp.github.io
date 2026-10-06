import sys, numpy as np
import approach as A
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from multiprocessing import Pool
def run(args):
    count, seed = args
    orig = A.GeneratedApproach._plan
    rec = []
    def P(self, r, h, tb, ob):
        orig(self, r, h, tb, ob)
        rec.append(len(self.wps) < 4 or self.wps[0][0][1] > 0.28)
    A.GeneratedApproach._plan = P
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
        count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
    s, info = env.reset(seed=seed, options={'object_count': count})
    ap = A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
    nexec = 0
    for t in range(300 + 150 * count):
        s, _, term, _, _ = env.step(ap.get_action(s))
        nexec += ap.phase == "exec"
        if ap.phase == "moves" or term: break
    return rec[:1], nexec
c = int(sys.argv[1])
with Pool(16) as p:
    res = p.map(run, [(c, s) for s in range(int(sys.argv[2]), int(sys.argv[3]))])
d = [n for r, n in res if r and r[0]]; f = [n for r, n in res if r and not r[0]]
print("direct", len(d), np.mean(d) if d else 0, "fallback", len(f), np.mean(f) if f else 0)
