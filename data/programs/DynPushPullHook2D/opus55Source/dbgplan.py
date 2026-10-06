import sys, numpy as np
import approach as A
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
orig = A.GeneratedApproach._plan
def P(self, r, h, tb, ob):
    orig(self, r, h, tb, ob)
    tl,tr,tbm,tt=self._extent(tb)
    print("PLAN sweep", self.sweep_dir, "tb", round(tl,2),round(tbm,2), round(tt,2), "wps", [tuple(round(float(v),2) for v in wp[0]) for wp in self.wps])
    objs=[o for o in [tb]+list(ob) if self._extent(o)[3]>A.WALL_Y]
    bx,_,th,_=self.wps[-1][0]
    ys=[]
    for y in np.linspace(0.27,1.45,6):
        P_,C=self._hook_pts(bx,y,th); ys.append((round(y,2),round(C[0],2),round(C[1],2),self._hook_clear(bx,y,th,objs)))
    print(ys)
    for o in objs: print([round(v,2) for v in self._extent(o)])
A.GeneratedApproach._plan = P
count=int(sys.argv[1]); seed=int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
s, info = env.reset(seed=seed, options={'object_count': count})
ap = A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
for t in range(400):
    s,_,term,_,_=env.step(ap.get_action(s))
    if ap.phase=="moves" or term: break
tb=None
