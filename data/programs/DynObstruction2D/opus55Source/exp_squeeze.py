import sys
from exp_common import *


def run(n, seed, name, gap=0.12, L=0.48, vdesc=0.01, tip_final=0.115, verbose=False, init_mod=None, tilt=0.0):
    env, state, ap = make(n, seed)
    if init_mod:
        state = set_block(env, state, name, **init_mod)
    S = Sim(env, state, ap)
    b = S.P()[1][name]
    gl, gr = wall_gaps(b)
    d = -1.0 if gr < gl else 1.0
    base_x = (WALL_R - 0.245) if d < 0 else 0.245
    th = -np.pi / 2 - d * tilt
    S.go(ty=1.45, tarm=0.24, tgap=gap)
    S.go(ty=1.45, tth=th, tarm=0.24, tgap=gap)
    S.go(tx=base_x, ty=1.45, tth=th, tarm=0.24, tgap=gap)
    S.go(tx=base_x, ty=1.45, tth=th, tarm=L, tgap=gap)
    def pb(tag):
        b = S.P()[1][name]
        print(tag, S.robot_str(), "B(%.3f,%.3f,th=%.3f)" % (b["x"], b["y"], b["theta"]), shapes_bb(env, n)['_left_finger_shape'][1])
    y_final = tip_final + L + 0.23
    for k in range(400):
        r = S.P()[0]
        if abs(r["y"] - y_final) < 1e-3: break
        S.step(ap._action(r, tx=base_x, ty=max(y_final, r["y"] - vdesc), tth=th))
        if verbose and k % 3 == 0: pb("d%d" % k)
    if verbose: pb("bottom")
    S.go(ty=1.45, n=60)
    for k in range(20): S.step(np.zeros(5))
    b = S.P()[1][name]
    g0 = min(gl, gr); g1 = min(wall_gaps(b))
    return g0, g1, b


if __name__ == "__main__":
    n, seed, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    kw = dict(gap=float(sys.argv[4]), L=float(sys.argv[5]), vdesc=float(sys.argv[6]), verbose=True)
    g0, g1, b = run(n, seed, name, **kw)
    print("gap %.3f -> %.3f" % (g0, g1), {k: round(v, 3) for k, v in b.items()})
