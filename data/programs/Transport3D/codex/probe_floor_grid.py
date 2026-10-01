"""Dense planar grasp scan for one commanded arm posture."""
import sys
import numpy as np
from env_client import make_env


q = np.asarray([float(x) for x in sys.argv[1].split(",")], dtype=float)
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
target = sys.argv[3] if len(sys.argv) > 3 else "box0"
spacing = float(sys.argv[4]) if len(sys.argv) > 4 else .08
radius = float(sys.argv[5]) if len(sys.argv) > 5 else .8
env = make_env()
s, _ = env.reset(seed=seed)


def g(name, feat):
    return float(s.get(s.get_object_from_name(name), feat))


def step(base=None, grip=0.0):
    global s
    a = np.zeros(11, np.float32)
    if base is not None:
        a[0] = np.clip(base[0] - g("robot", "pos_base_x"), -.2, .2)
        a[1] = np.clip(base[1] - g("robot", "pos_base_y"), -.2, .2)
    current = np.asarray([g("robot", "joint_%d" % i) for i in range(1, 8)])
    a[3:10] = np.clip(q - current, -.2, .2)
    a[10] = grip
    s, _, terminated, truncated, _ = env.step(a)
    return terminated or truncated


tx, ty = g(target, "pose_x"), g(target, "pose_y")
for _ in range(32):
    step(grip=1.)

# Broad planar scan. Every point gets an open state followed by a close edge.
points = np.arange(-radius, radius + spacing / 2, spacing)
for iy, oy in enumerate(points):
    xs = points if iy % 2 == 0 else points[::-1]
    for ox in xs:
        goal = (tx + ox, ty + oy)
        step(goal, 1.)
        ended = step(goal, -1.)
        if g("robot", "grasp_active") > .5:
            actual_q = [g("robot", "joint_%d" % i) for i in range(1, 8)]
            print("HIT", "q", actual_q, "object-minus-base",
                  [tx-g("robot", "pos_base_x"), ty-g("robot", "pos_base_y")],
                  "base-rotation", g("robot", "pos_base_rot"),
                  "tf", [g("robot", "grasp_tf_%s" % c) for c in "xyz"])
            env.close()
            raise SystemExit
        if ended:
            print("ENDED")
            env.close()
            raise SystemExit
print("MISS", q.tolist(), "final object", [g(target, "pose_%s" % c) for c in "xyz"])
env.close()
