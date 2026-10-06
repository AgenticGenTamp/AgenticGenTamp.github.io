import sys, json
from exp_common import *
from exp_squeeze import run
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
cases = [(int(a), int(b), c, None) for a, b, c in (l.split() for l in NAT.splitlines())]
# artificial: target block flush against a wall with various sizes (count 0)
rng = np.random.default_rng(0)
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 0):
    w, h = rng.uniform(0.16, 0.48), rng.uniform(0.24, 0.48)
    g = rng.uniform(0.0, 0.2)
    side = rng.choice([-1, 1])
    x = (w / 2 + g) if side < 0 else (WALL_R - w / 2 - g)
    cases.append((0, 100 + i, "target_block", dict(x=x, width=w, height=h, y=0.1 + h / 2)))
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
ok = 0
for n, seed, name, mod in cases:
    g0, g1, b = run(n, seed, name, init_mod=mod, **kw)
    th = b["theta"] % (np.pi / 2)
    th = min(th, np.pi / 2 - th)
    hh = b["height"] / 2 if abs(np.cos(b["theta"])) > 0.7 else b["width"] / 2
    flat = th < 0.05 and abs(b["y"] - (0.1 + hh)) < 0.01
    good = flat and g1 >= 0.30
    ok += good
    print(f"{n} {seed} {name:13s} w={b['width']:.2f} h={b['height']:.2f} gap {g0:.3f}->{g1:.3f} th={b['theta']:.3f} y={b['y']:.3f} {'OK' if good else 'FAIL'}", flush=True)
print("success", ok, "/", len(cases))
