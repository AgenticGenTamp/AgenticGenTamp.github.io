"""Focused seed-3 diagonal can push scan via joint-1 yaw offsets."""

import numpy as np
from env_client import make_env


env = make_env()
for dq1 in (.24, .27, .32):
    s, _ = env.reset(seed=3); home = s[96:103].copy()
    # Folded lateral alignment below the can.
    goal_y = float(s[33]) - .28
    for _ in range(35):
        a = np.zeros(11, np.float32)
        a[1] = np.clip((goal_y-float(s[94]))*.5, -.1, .1)
        s, *_ = env.step(a)
        if abs(goal_y-float(s[94])) < .025: break
    pose = home.copy(); pose[0] += dq1; pose[1] = 1.0; pose[3] = -1.56
    for _ in range(80):
        err = pose-s[96:103]
        if np.max(np.abs(err)) < .035: break
        a = np.zeros(11, np.float32); a[3:10] = np.clip(err*.5, -.1, .1)
        s, *_ = env.step(a)
    before = s.copy(); moved = False
    for k in range(25):
        a = np.zeros(11, np.float32); a[1] = .02
        s, r, term, trunc, info = env.step(a)
        if np.linalg.norm(s[32:34]-before[32:34]) > .003: moved = True
        if term or trunc: break
    # settle briefly
    for _ in range(5): s, r, term, trunc, info = env.step(np.zeros(11,np.float32))
    print("dq1",dq1,"base",np.round(before[93:95],3).tolist(),
          "q1",round(float(before[96]),3),"moved",moved,
          "can0",np.round(before[32:35],4).tolist(),
          "can1",np.round(s[32:35],4).tolist(),
          "dxy",np.round(s[32:34]-before[32:34],4).tolist(),
          "tiltxy",np.round(s[36:38],4).tolist(),
          "bowlxy",np.round(s[0:2],4).tolist(),"r",r,flush=True)
env.close()
