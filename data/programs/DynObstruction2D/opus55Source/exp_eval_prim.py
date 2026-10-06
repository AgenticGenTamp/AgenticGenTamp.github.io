import sys
from exp_squeeze2 import get_env, cases, judge, set_block, wall_gaps, np
from exp_primitive import WallUnjam
n_art = int(sys.argv[1]) if len(sys.argv) > 1 else 16
cseed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cs = cases(n_art, cseed)
if cseed != 0: cs = cs[24:]
ok = 0
for n, seed, name, mod in cs:
    env = get_env(n)
    state, _ = env.reset(seed=seed, options={"object_count": n})
    if mod: state = set_block(env, state, name, **mod)
    from exp_common import blocks
    b0 = blocks(state)
    g0 = min(wall_gaps(b0[name]))
    P = WallUnjam(env.action_space, name)
    T = 0
    while T < 1500:
        a = P.step(state)
        if a is None: break
        state, *_ = env.step(a); T += 1
    b1 = blocks(state)
    b = b1[name]; g1 = min(wall_gaps(b))
    others = max([np.hypot(b1[k]['x'] - b0[k]['x'], b1[k]['y'] - b0[k]['y']) for k in b0 if k != name] + [0])
    good = judge(b, g1); ok += good
    print(f"{n} {seed} {name:13s} w={b['width']:.2f} h={b['height']:.2f} gap {g0:.3f}->{g1:.3f} th={b['theta']:.3f} y={b['y']:.3f} cfg={P.ci} T={T} others_moved={others:.2f} {'OK' if good else 'FAIL'}", flush=True)
print("success", ok, "/", len(cs))
