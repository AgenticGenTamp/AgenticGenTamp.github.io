import sys
from exp_common import *
n, seed, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
pen = float(sys.argv[4]); off = float(sys.argv[5]); gap = float(sys.argv[6]) if len(sys.argv) > 6 else 0.12
vx = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05
env, state, ap = make(n, seed)
S = Sim(env, state, ap)
b = S.P()[1][name]
gl, gr = wall_gaps(b)
d = -1.0 if gr < gl else 1.0   # direction away from wall
bl, br = xext(b)
top = b["y"] + b["height"] / 2
far = bl if d < 0 else br
# far-side finger centre at far edge - d*off
fx = far - d * off
bx = fx - d * gap / 2
print("block", b, "d", d)
S.go(ty=1.45, tarm=0.24, tgap=gap, tth=None)
S.go(ty=1.45, tth=-np.pi / 2, tarm=0.24, tgap=gap)
S.go(tx=bx, ty=1.45, tth=-np.pi / 2, tarm=0.24, tgap=gap)
ty_touch = top + 0.24 + 0.23
S.go(tx=bx, ty=ty_touch + 0.03, tth=-np.pi / 2, tarm=0.24, tgap=gap)
for k in range(200):
    r = S.P()[0]
    tgt = ty_touch - pen
    if abs(r["y"] - tgt) < 1e-3: break
    S.step(ap._action(r, tx=bx, ty=max(tgt, r["y"] - 0.003)))
b = S.P()[1][name]
print("pressed", S.robot_str(), {k: round(v, 3) for k, v in b.items()})
for k in range(40):
    r = S.P()[0]
    S.step(ap._action(r, tx=r["x"] + d * vx, ty=ty_touch - pen))
    b = S.P()[1][name]
    if k % 3 == 0: print(k, S.robot_str(), {k2: round(v, 3) for k2, v in b.items()})
S.go(ty=1.45, n=60)
for k in range(30): S.step(np.zeros(5))
b = S.P()[1][name]
print("final", {k: round(v, 3) for k, v in b.items()}, "gaps", np.round(wall_gaps(b), 3))
