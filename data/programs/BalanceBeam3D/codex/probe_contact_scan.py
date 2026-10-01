"""Broad scan for arm/cube contact at a fixed candidate pickup posture."""
import json
import numpy as np

from env_client import make_env


env = make_env()
s, _ = env.reset(seed=0)
initial = s.copy()
cube = initial[:3].copy()
qtarget = np.array([0, 1.58, np.pi, -1.46, 0, -0.872665, np.pi / 2])

# First reach the posture well away from all cubes.
for _ in range(110):
    a = np.zeros(11, np.float32)
    a[:2] = np.clip(1.2 * (np.array([-0.25, -0.6]) - s[16:18]), -.1, .1)
    a[3:10] = np.clip(1.8 * (qtarget - s[19:26]), -.1, .1)
    a[10] = 0.0
    s, r, te, tr, info = env.step(a)

found = False
for dx in np.arange(-0.45, 0.66, 0.1):
    ys = np.arange(-0.55, 0.56, 0.1)
    if int(round((dx + .45) / .1)) % 2:
        ys = ys[::-1]
    for dy in ys:
        bt = cube[:2] - np.array([dx, dy])
        for _ in range(4):
            a = np.zeros(11, np.float32)
            a[:2] = np.clip(1.2 * (bt - s[16:18]), -.1, .1)
            a[3:10] = np.clip(1.8 * (qtarget - s[19:26]), -.1, .1)
            a[10] = 0.0
            s, r, te, tr, info = env.step(a)
        delta = np.r_[s[:3] - initial[:3], s[54:57] - initial[54:57],
                      s[70:73] - initial[70:73]]
        if np.max(np.abs(delta)) > .003:
            print("CONTACT", round(float(dx), 3), round(float(dy), 3),
                  np.round(s[16:19], 4).tolist(), np.round(delta, 4).tolist())
            print(json.dumps(s.tolist()))
            found = True
            break
    if found:
        break
if not found:
    print("NO_CONTACT", np.round(s[19:26], 4).tolist())
env.close()
