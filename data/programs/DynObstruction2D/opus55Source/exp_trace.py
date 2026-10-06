import sys, json
from exp_squeeze3 import *
from exp_common import xext
import exp_squeeze3 as E
cs = cases(16)
sd = int(sys.argv[1]); kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
c = [c for c in cs if c[1] == sd and (len(sys.argv) < 4 or c[2] == sys.argv[3])][0]
n, seed, name, mod = c
env = get_env(n)
state, _ = env.reset(seed=seed, options={"object_count": n})
if mod: state = set_block(env, state, name, **mod)
S = Sim(env, state, Ctl(env.action_space))
orig = S.step
def step(a):
    orig(a)
    r = S.P()[0]; b = S.P()[1][name]
    if r["y"] < 1.2:
        P = robot_polys(r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"])
        fp, fm = P["fp"], P["fm"]
        print("t%d R y=%.3f fp x[%.3f,%.3f] ymin %.3f | fm x[%.3f,%.3f] ymin %.3f | B x[%.3f,%.3f] y=%.3f top=%.3f th=%.3f" % (
            S.t, r["y"], fp[:,0].min(), fp[:,0].max(), fp[:,1].min(), fm[:,0].min(), fm[:,0].max(), fm[:,1].min(),
            *xext(b), b["y"], b["y"] + b["height"]/2, b["theta"]))
S.step = step
for a in kw.pop("tilts", [0.4]):
    print(attempt(S, name, a, verbose=True, **kw))
