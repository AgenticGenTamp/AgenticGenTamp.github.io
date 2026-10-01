"""Fine grasp grid around a URDF-derived ground-level tool pose."""
import numpy as np
from env_client import make_env
from probe_grasp_sequence import drive, pose, value

GROUND = np.array([-.12, 2.24, 3.1416, -.20, 0.0, -.83, 1.5708])

env = make_env()
state, _ = env.reset(seed=0, options={"object_count": 1})
p0 = pose(state, "cube1")
state = drive(env, state, GROUND, 0.0, 150)
robot0 = np.array([value(state, "robot", "pos_base_x"), value(state, "robot", "pos_base_y")])
found = False
for ix, dx in enumerate(np.arange(-.03, .091, .01)):
    ys = np.arange(-.06, .061, .01)
    if ix % 2: ys = ys[::-1]
    for dy in ys:
        # Closed-loop translate to each fine-grid point.
        for _ in range(8):
            now = np.array([value(state, "robot", "pos_base_x"), value(state, "robot", "pos_base_y")])
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(2.0 * (robot0 + [dx, dy] - now), -.1, .1)
            a[10] = 0.0
            state, *_ = env.step(a)
        for grip in (1.0, 1.0, 1.0):
            a = np.zeros(11, np.float32); a[10] = grip
            state, *_ = env.step(a)
        # A 2 cm jiggle reveals attachment while remaining inside the grid.
        a = np.zeros(11, np.float32); a[0] = .025; a[10] = 1.0
        state, *_ = env.step(a)
        moved = np.linalg.norm(pose(state, "cube1") - p0)
        if moved > .001:
            print("FOUND", round(dx,3), round(float(dy),3), "moved", moved,
                  "cube", pose(state,"cube1")); found = True; break
        a[:] = 0.; a[10] = 0.; state, *_ = env.step(a)
    if found: break
print("done found", found, "cube_delta", pose(state,"cube1")-p0)
env.close()
