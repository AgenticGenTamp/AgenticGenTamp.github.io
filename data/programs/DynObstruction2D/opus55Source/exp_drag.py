import sys
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach

n, seed = 0, 0
pen = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
env = VariableObjectCountEnv(
    constant_object_env_path="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction",
    design_counts=[n], eval_counts=[n])
state, info = env.reset(seed=seed, options={"object_count": n})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(state, info)


def P(state):
    return ap._parse(state)


def act(r, **kw):
    return ap._action(r, **kw)


def step(a):
    global state
    state, *_ = env.step(a)


r, b, s, _ = P(state)
# go up, rotate, close gripper
for _ in range(60):
    r, b, s, _ = P(state)
    step(act(r, tx=max(r["x"], 0.8), ty=1.45, tth=-np.pi / 2, tarm=0.24, tgap=0.12))
# travel above block's centre-left
top = b["y"] + b["height"] / 2
tx = b["x"] - 0.4
for _ in range(80):
    r, b, s, _ = P(state)
    step(act(r, tx=tx, ty=1.45, tth=-np.pi / 2, tarm=0.24, tgap=0.12))
# descend: tip y = base_y - L - 0.2 (approx); want tip at top - pen
for L in [0.24]:
    ty = top - pen + L + 0.2
for k in range(60):
    r, b, s, _ = P(state)
    step(act(r, tx=tx, ty=max(ty, r['y']-0.05 if r['y']>ty+0.04 else r['y']-0.004), tth=-np.pi / 2, tarm=0.24, tgap=0.12))
    r, b, s, _ = P(state)
    if k > 5: print('desc', k, f"R {r['x']:.3f} {r['y']:.3f} B {b['x']:.3f} {b['y']:.3f}")
r, b, s, _ = P(state)
for i in range(20):
    r, b, s, _ = P(state)
    step(act(r, tx=r["x"] + 0.05, ty=ty, tth=-np.pi / 2, tarm=0.24, tgap=0.12))
    r, b, s, _ = P(state)
    print('pre', i, f"R {r['x']:.3f} {r['y']:.3f} B {b['x']:.3f} {b['y']:.3f} th {b['theta']:.3f}")
print("after descend R", r["x"], r["y"], "B", b["x"], b["y"], "top", top)
for i in range(0):
    r, b, s, _ = P(state)
    step(act(r, tx=r["x"] - 0.05, ty=ty, tth=-np.pi / 2, tarm=0.24, tgap=0.12))
    r, b, s, _ = P(state)
    print(i, f"R {r['x']:.3f} {r['y']:.3f} B {b['x']:.3f} {b['y']:.3f} th {b['theta']:.3f}")
