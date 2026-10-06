import sys, json
from exp_common import *
from exp_geom import robot_polys

_ENVS = {}


def get_env(n):
    if n not in _ENVS:
        _ENVS[n] = make(n, 0)[0]
    return _ENVS[n]


def place_x(y, th, L, gap, d, margin=0.012):
    """base x so that robot parts are as close to the wall (side -d) as allowed."""
    # extent of parts relative to base x
    P = robot_polys(0.0, y, th, L, gap)
    allx = np.concatenate([P[k][:, 0] for k in ("plate", "fp", "fm")])
    if d < 0:  # right wall
        return min(WALL_R - margin - allx.max(), WALL_R - 0.245)
    return max(margin - allx.min(), 0.245)


def run(n, seed, name, gap=0.12, L=0.48, a=0.0, vdesc=0.01, tip_final=0.12, verbose=False,
        init_mod=None, extra=0.0, push_steps=0, push_v=0.02, g_target=0.45):
    env = get_env(n)
    state, _ = env.reset(seed=seed, options={"object_count": n})
    ap = Ctl(env.action_space)
    if init_mod:
        state = set_block(env, state, name, **init_mod)
    S = Sim(env, state, ap)
    b = S.P()[1][name]
    gl, gr = wall_gaps(b)
    d = -1.0 if gr < gl else 1.0
    th = -np.pi / 2 - d * a
    base_x = place_x(1.0, th, L, gap, d) + d * extra
    S.go(ty=1.45, tarm=0.24, tgap=gap)
    S.go(ty=1.45, tth=th, tarm=0.24, tgap=gap)
    S.go(tx=base_x, ty=1.45, tth=th, tarm=0.24, tgap=gap)
    S.go(tx=base_x, ty=1.45, tth=th, tarm=L, tgap=gap)
    r = S.P()[0]
    P = robot_polys(r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"])
    low = min(P["fp"][:, 1].min(), P["fm"][:, 1].min())
    y_final = r["y"] - (low - tip_final)
    def pb(tag):
        b = S.P()[1][name]
        print(tag, S.robot_str(), "B(%.3f,%.3f,th=%.3f)" % (b["x"], b["y"], b["theta"]))
    for k in range(600):
        r = S.P()[0]
        if abs(r["y"] - y_final) < 1e-3: break
        S.step(ap._action(r, tx=base_x, ty=max(y_final, r["y"] - vdesc), tth=th, tarm=L, tgap=gap))
        if verbose and k % 3 == 0: pb("d%d" % k)
    if verbose: pb("bottom")
    for k in range(int(push_steps)):
        b = S.P()[1][name]
        if min(wall_gaps(b)) >= g_target: break
        r = S.P()[0]
        S.step(ap._action(r, tx=r["x"] + d * push_v, ty=y_final, tth=th, tarm=L, tgap=gap))
        if verbose and k % 2 == 0: pb("p%d" % k)
    S.go(ty=1.45, n=60)
    for k in range(20): S.step(np.zeros(5))
    b = S.P()[1][name]
    return min(gl, gr), min(wall_gaps(b)), b


NAT = """0 0 target_block
0 1 target_block
0 16 target_block
0 18 target_block
0 19 target_block
1 23 obstruction0
1 29 obstruction0
1 31 obstruction0
2 1 obstruction0
2 5 target_block
2 10 obstruction0
2 12 obstruction1
2 28 obstruction1
2 32 obstruction1
2 35 target_block
2 37 target_block
3 5 obstruction0
3 7 obstruction0
3 9 target_block
3 10 obstruction0
3 14 obstruction0
3 17 target_block
3 23 obstruction2
3 25 obstruction2"""


def cases(n_art=16, seed=0):
    cs = [(int(a), int(b), c, None) for a, b, c in (l.split() for l in NAT.splitlines())]
    rng = np.random.default_rng(seed)
    for i in range(n_art):
        w, h = rng.uniform(0.16, 0.48), rng.uniform(0.24, 0.48)
        g = rng.uniform(0.0, 0.2)
        side = rng.choice([-1, 1])
        x = (w / 2 + g) if side < 0 else (WALL_R - w / 2 - g)
        cs.append((0, 100 + i, "target_block", dict(x=x, width=w, height=h, y=0.1 + h / 2)))
    return cs


def judge(b, g1, need=0.30):
    th = b["theta"] % (np.pi / 2)
    th = min(th, np.pi / 2 - th)
    hh = b["height"] / 2 if abs(np.cos(b["theta"])) > 0.7 else b["width"] / 2
    flat = th < 0.05 and abs(b["y"] - (0.1 + hh)) < 0.01
    return flat and g1 >= need


if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    n_art = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    ok = 0; cs = cases(n_art)
    for n, seed, name, mod in cs:
        g0, g1, b = run(n, seed, name, init_mod=mod, **kw)
        good = judge(b, g1); ok += good
        print(f"{n} {seed} {name:13s} w={b['width']:.2f} h={b['height']:.2f} gap {g0:.3f}->{g1:.3f} th={b['theta']:.3f} y={b['y']:.3f} {'OK' if good else 'FAIL'}", flush=True)
    print("success", ok, "/", len(cs), kw)
