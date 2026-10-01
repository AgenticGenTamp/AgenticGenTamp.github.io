import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
for gap in [0.004, 0.008, 0.012, 0.016, 0.02, 0.03]:
    obs, info = env.reset(seed=2104, options={'object_count': 4})
    obs, *_ = env.step(np.array([0,0,0,0.1,0], dtype=np.float32))
    ap._parse(obs); o = ap.obst['obstruction0']
    tx = o['x'] + o['w']/2
    ty = o['y'] + o['h'] + gap + 0.005 + 0.2
    for _ in range(60):
        ap._parse(obs); rx, ry = ap.robot['x'], ap.robot['y']
        if abs(tx-rx) < 1e-5 and abs(ty-ry) < 1e-5: break
        dx, dy = np.clip(tx-rx, -0.05, 0.05), np.clip(ty-ry, -0.05, 0.05)
        obs, *_ = env.step(np.array([dx,dy,0,0,0], dtype=np.float32))
    obs, *_ = env.step(np.array([0,0,0,0,1], dtype=np.float32))
    obs, *_ = env.step(np.array([0,0.05,0,0,1], dtype=np.float32))
    ap._parse(obs)
    print('gap', gap, 'robot y', round(ap.robot['y']-ty,4), 'obj y moved', round(ap.obst['obstruction0']['y']-o['y'],4))
