"""Raster the base under random static arm configurations in one episode."""
import numpy as np
from env_client import make_env


def v(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in fs])


env = make_env()
s, _ = env.reset(seed=3, options={"object_count": 1})
c0 = v(s, "cube_0", ("x", "y", "z"))
rng = np.random.default_rng(448)
qs = []
for _ in range(12):
    qs.append(np.array([rng.uniform(-.6, .6), rng.uniform(-.6, 1.1),
                        np.pi + rng.uniform(-.4, .4), rng.uniform(-2.7, .1),
                        rng.uniform(-.8, .8), rng.uniform(-2.5, .1),
                        rng.uniform(.5, 2.7)]))
found = False
for qi, q in enumerate(qs):
    # snake grid; offsets are cube minus base
    points = []
    for jj, yo in enumerate(np.linspace(-.42, .42, 7)):
        xx = np.linspace(.12, .88, 7)
        if jj % 2:
            xx = xx[::-1]
        points.extend((x, yo) for x in xx)
    for pi, (xo, yo) in enumerate(points):
        goal = np.array([c0[0] - xo, c0[1] - yo, 0.0])
        for k in range(5 if pi else 22):
            b = v(s, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
            cq = v(s, "robot", tuple("pos_arm_joint%d" % z for z in range(1, 8)))
            a = np.zeros(18, np.float32)
            a[:3] = np.clip(goal-b, -.1, .1)
            a[3:10] = np.clip(q-cq, -.1, .1)
            a[11:18] = np.clip(4*(q-cq), -2.5, 2.5)
            # Toggle periodically: catches proximity welds of either polarity.
            a[10] = float((pi // 2) % 2)
            s, _, term, trunc, _ = env.step(a)
            c = v(s, "cube_0", ("x", "y", "z"))
            if np.linalg.norm(c-c0) > .002:
                print("FOUND qi", qi, "point", pi, "step", k,
                      "offset", (xo,yo), "grip", a[10],
                      "qtarget", np.round(q,4), "qactual", np.round(cq,4),
                      "cube", np.round(c0,4), np.round(c,4), flush=True)
                found = True
                break
            if term or trunc:
                break
        if found or term or trunc:
            break
    print("done q", qi, flush=True)
    if found or term or trunc:
        break
if not found:
    print("NONE final", np.round(v(s,"cube_0",("x","y","z")),4))
env.close()
