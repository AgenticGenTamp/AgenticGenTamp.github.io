"""Close before entering the known wiper-contact pose, then tug north."""
import math
import numpy as np
from env_client import make_env


def v(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def xy(s, n):
    fs = ("pos_base_x", "pos_base_y") if n == "robot" else ("x", "y")
    return np.array([v(s, n, f) for f in fs])


def q(s):
    return np.array([v(s, "robot", "pos_arm_joint%d" % i) for i in range(1, 8)])


def ae(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


e = make_env()
# First test the opposite polarity: the known contact trajectory with command 1
# retained during the tug. Earlier probes always changed to 0 before tugging.
s, _ = e.reset(seed=0, options={"object_count": 1})
w0 = xy(s, "wiper_0")
final = np.array([1.3378, 1.0894])
qt = np.array([0, .8, 2.37, -2.57, .02, -.8, 1.57])
for _ in range(45):
    a = np.zeros(11, np.float32)
    a[:2] = np.clip(.8 * (final - xy(s, "robot")), -.1, .1)
    a[2] = np.clip(.8 * ae(-1.475, v(s, "robot", "pos_base_rot")), -.1, .1)
    a[3:10] = np.clip(.7 * (qt - q(s)), -.1, .1); a[10] = 1
    s, *_ = e.step(a)
after = xy(s, "wiper_0").copy()
for _ in range(8):
    a = np.zeros(11, np.float32); a[1] = .06; a[10] = 1
    a[3:10] = np.clip(.5 * (qt - q(s)), -.1, .1); s, *_ = e.step(a)
print("KEEP_ONE", "contact", (after-w0).round(3),
      "tug", (xy(s,"wiper_0")-after).round(3), flush=True)

for clearance in (0.04, 0.07, 0.10):
    for dx in (-0.025, 0.0, 0.025):
        s, _ = e.reset(seed=0, options={"object_count": 1})
        w0 = xy(s, "wiper_0")
        final = np.array([1.3378 + dx, 1.0894])
        qt = np.array([0, .8, 2.37, -2.57, .02, -.8, 1.57])
        # Approach from north, still open.
        for _ in range(38):
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(.8 * (final + [0, clearance] - xy(s, "robot")), -.1, .1)
            a[2] = np.clip(.8 * ae(-1.475, v(s, "robot", "pos_base_rot")), -.1, .1)
            a[3:10] = np.clip(.7 * (qt - q(s)), -.1, .1)
            a[10] = 1
            s, *_ = e.step(a)
        preclose = xy(s, "wiper_0").copy()
        # Close in free space.
        for _ in range(8):
            a = np.zeros(11, np.float32); a[10] = 0
            s, *_ = e.step(a)
        # Ingress into handle with fingers closed.
        for _ in range(10):
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(.5 * (final - xy(s, "robot")), -.04, .04)
            a[3:10] = np.clip(.5 * (qt - q(s)), -.1, .1)
            a[10] = 0; s, *_ = e.step(a)
        after = xy(s, "wiper_0").copy()
        # Tug back north.
        for _ in range(8):
            a = np.zeros(11, np.float32); a[1] = .06; a[10] = 0
            a[3:10] = np.clip(.5 * (qt - q(s)), -.1, .1)
            s, *_ = e.step(a)
        print(clearance, dx, "pre", (preclose-w0).round(3),
              "ingress", (after-preclose).round(3), "tug", (xy(s,"wiper_0")-after).round(3),
              "total", (xy(s,"wiper_0")-w0).round(3), flush=True)
e.close()
