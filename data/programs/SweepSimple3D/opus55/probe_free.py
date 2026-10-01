"""Drive the bare base along rays from a center to map the free space boundary."""
import sys, numpy as np
from env_client import make_env
from rutil import Bot
cx, cy = float(sys.argv[1]), float(sys.argv[2]); angs = [float(a) for a in sys.argv[3].split(',')]
env = make_env(); obs, _ = env.reset(seed=1); b = Bot(env, obs)
for ang in angs:
    b.goto(base_t=[cx, cy, 0.0], tol=0.02, max_steps=150)
    t = np.deg2rad(ang); tgt = [cx + 4 * np.cos(t), cy + 4 * np.sin(t), 0.0]
    last = b.base(); stall = 0
    for k in range(120):
        b.act(base_t=tgt)
        p = b.base()
        if np.linalg.norm(p[:2] - last[:2]) < 0.003: stall += 1
        else: stall = 0
        last = p
        if stall > 5: break
    print('ang', ang, 'stop', np.round(b.base(), 3), flush=True)
