"""Sweep the 5 cubes off the near edge of the kitchen-island counter.

Technique: closed gripper, tool pointing down (RDOWN), tip at world z=0.466
(tip bottom right at the counter top, z~0.4598).  Position the tip ~9cm beyond
a cube (smaller world x), then BACK THE BASE UP in +x with the arm joints
frozen: the tip translates rigidly in +x world and drags the cube over the
near edge (world x ~= 0.89).  One cube per pass, ~65 steps per pass.
"""
import numpy as np
from env_client import make_env
from ctrl2 import moveto, move_base, RDOWN, tip_to_fk, JLO, JHI
from ik import ik

ZSW   = 0.466   # tip z while sweeping (commanded; true tip ~4mm lower)
ZHI   = 0.58    # safe transit height
EDGE  = 0.89    # counter near edge, world x
DRAG  = 0.42    # base back-up distance per pass
GRIP  = 0.0     # closed

def w2l(base, wx, wy):
    bx, by, yaw = base
    dx, dy = wx-bx, wy-by
    return np.cos(yaw)*dx+np.sin(yaw)*dy, -np.sin(yaw)*dx+np.cos(yaw)*dy

def tipw(base, wx, wy, wz):
    lx, ly = w2l(base, wx, wy)
    return [lx, ly, wz]

def cubes(o):
    return np.asarray(o[0:80]).reshape(5, 16)[:, 0:3]

def q_for(base, q_cur, wp):
    """joint target that puts the tip at world point wp when the base is at `base`"""
    p = tip_to_fk(tipw(base, *wp), RDOWN)
    q, res = ik(np.array(p), RDOWN, np.asarray(q_cur).copy())
    return np.clip(q, JLO, JHI), res

def sweep(env, o, verbose=True):
    o = np.asarray(o, float)
    HOME = [float(o[125]), float(o[126]), float(o[127])]
    total = 0
    for it in range(7):
        c = cubes(o)
        on = [j for j in range(5) if c[j][2] > 0.40 and c[j][0] < EDGE+0.03]
        if not on:
            break
        j = max(on, key=lambda k: c[k][0])          # push the closest-to-edge cube first
        cx, cy = float(c[j][0]), float(c[j][1])
        sx = min(cx-0.085, 0.78)                    # tip start, beyond the cube
        # 1) transit: raise arm, then drive base home while the arm pre-positions
        o, d = moveto(env, o, tipw(o[125:128], o[125]-0.35, cy, ZHI), RDOWN,
                      steps=14, grip=GRIP); total += d['used']
        qhi, _ = q_for(HOME, o[128:135], (sx, cy, ZHI))
        n = 18 if it else 6
        o = move_base(env, o, HOME, steps=n, grip=GRIP, qhold=qhi); total += n
        # 2) descend onto the counter behind the cube
        o, d = moveto(env, o, tipw(o[125:128], sx, cy, ZSW), RDOWN,
                      steps=45, grip=GRIP); total += d['used']
        # 3) freeze the arm, back the base up -> tip sweeps +x
        q = o[128:135].copy()
        o = move_base(env, o, [o[125]+DRAG, HOME[1], HOME[2]], steps=10,
                      grip=GRIP, qhold=q); total += 10
        if verbose:
            print(f"pass{it} cube{j} start=({sx:.3f},{cy:.3f}) "
                  f"now={np.round(cubes(o)[j],3)} steps={total}")
    return o, total

if __name__ == "__main__":
    env = make_env()
    o, _ = env.reset(seed=0); o = np.asarray(o, float)
    print("cubes0\n", np.round(cubes(o), 3))
    o, total = sweep(env, o)
    c = cubes(o)
    print("final cubes\n", np.round(c, 3))
    print("off-counter:", sum(1 for j in range(5) if c[j][2] < 0.30), "/5  steps:", total)
    print("wiper", np.round(np.asarray(o[147:150]), 3), "base", np.round(o[125:128], 3))
    env.close()
