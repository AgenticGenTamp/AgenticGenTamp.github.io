import sys, json
from exp_squeeze2 import get_env, place_x, cases, judge, Ctl, Sim, set_block, wall_gaps, robot_polys, np, WALL_R


def rest_y(b):
    return 0.1 + (b["height"] / 2 if abs(np.cos(b["theta"])) > 0.7 else b["width"] / 2)


def attempt(S, name, a, gap=0.12, L=0.48, vdesc=0.0499, tip_final=0.12, sink_tol=0.015,
            push_steps=30, push_v=0.02, g_target=0.45, verbose=False):
    ap = S.ap
    b = S.P()[1][name]
    gl, gr = wall_gaps(b)
    d = -1.0 if gr < gl else 1.0
    th = -np.pi / 2 - d * a
    base_x = place_x(1.0, th, L, gap, d)
    S.go(ty=1.45, tarm=0.24, tgap=gap)
    S.go(ty=1.45, tth=th, tarm=0.24, tgap=gap)
    S.go(tx=base_x, ty=1.45, tth=th, tarm=0.24, tgap=gap)
    S.go(tx=base_x, ty=1.45, tth=th, tarm=L, tgap=gap)
    r = S.P()[0]
    P = robot_polys(r["x"], r["y"], r["theta"], r["arm_joint"], r["finger_gap"])
    low = min(P["fp"][:, 1].min(), P["fm"][:, 1].min())
    y_final = r["y"] - (low - tip_final)
    status = "ok"
    for k in range(600):
        r = S.P()[0]
        if abs(r["y"] - y_final) < 1e-3: break
        S.step(ap._action(r, tx=base_x, ty=max(y_final, r["y"] - vdesc), tth=th, tarm=L, tgap=gap))
        b = S.P()[1][name]
        if b["y"] < rest_y(b) - sink_tol:
            status = "sink"; break
    if status == "ok":
        for k in range(int(push_steps)):
            b = S.P()[1][name]
            if min(wall_gaps(b)) >= g_target: break
            r = S.P()[0]
            S.step(ap._action(r, tx=r["x"] + d * push_v, ty=y_final, tth=th, tarm=L, tgap=gap))
    for k in range(80):
        r = S.P()[0]
        if r["y"] > 1.44: break
        S.step(ap._action(r, ty=1.45))
    for k in range(10): S.step(np.zeros(5))
    b = S.P()[1][name]
    if verbose: print("  attempt a=%.2f %s gap->%.3f y=%.3f th=%.3f" % (a, status, min(wall_gaps(b)), b["y"], b["theta"]))
    return status


def run(n, seed, name, init_mod=None, tilts=(0.4, 0.5, 0.3, 0.6), need=0.30, verbose=False, **kw):
    env = get_env(n)
    state, _ = env.reset(seed=seed, options={"object_count": n})
    if init_mod:
        state = set_block(env, state, name, **init_mod)
    S = Sim(env, state, Ctl(env.action_space))
    b = S.P()[1][name]
    g0 = min(wall_gaps(b))
    tries = 0
    for a in tilts:
        tries += 1
        attempt(S, name, a, verbose=verbose, **kw)
        b = S.P()[1][name]
        if judge(b, min(wall_gaps(b)), need):
            break
        if abs(b["y"]) > 3: break
    return g0, min(wall_gaps(b)), b, tries, S.t


if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    n_art = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    cseed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ok = 0; cs = cases(n_art, cseed)
    if cseed != 0: cs = cs[24:]
    for n, seed, name, mod in cs:
        g0, g1, b, tries, T = run(n, seed, name, init_mod=mod, verbose=True, **kw)
        good = judge(b, g1); ok += good
        print(f"{n} {seed} {name:13s} w={b['width']:.2f} h={b['height']:.2f} gap {g0:.3f}->{g1:.3f} th={b['theta']:.3f} y={b['y']:.3f} tries={tries} T={T} {'OK' if good else 'FAIL'}", flush=True)
    print("success", ok, "/", len(cs), kw)
