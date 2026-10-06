exec(open('dbg.py').read().split("ap = Gen")[0])
import approach
orig = approach.Sim.navigate
def nav(self, x, y, th):
    print("navigate from", np.round(self.base(),3), "to", np.round([x,y,th],3), "route", [np.round(w,2) for w in self.route(self.base()[:2], np.array([x,y]))])
    try:
        return orig(self, x, y, th)
    except approach.PlanFail as e:
        print("  fail", e, np.round(self.base(),3)); raise
approach.Sim.navigate = nav
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
sim=ap.sims[1]; sim.load(s); import approach as A
print(repr(ap.sims[1].base()))
