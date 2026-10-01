from env_client import make_env
import numpy as np
env = make_env()
for direction in ["x", "y", "-x"]:
    obs, _ = env.reset(seed=0)
    b = obs[54:57].copy()  # sb1 at (0.703, 0.206)
    # position base: for x-approach, align y and go from far; for y: align x and start from y-1.0
    if direction == "x": start = np.array([b[0]-1.0, b[1]]); d = np.array([1,0])
    elif direction == "y": start = np.array([b[0], b[1]+1.0]); d = np.array([0,-1])
    else: start = np.array([b[0]+1.0, b[1]]); d = np.array([-1,0])
    # go to start via a detour avoiding (move y first far away)
    for goal in [np.array([obs[16], start[1] + (0.6 if direction!="y" else 0)]), np.array([start[0], start[1] + (0.6 if direction!="y" else 0)]), start]:
        for t in range(60):
            a = np.zeros(11, np.float32); a[:2] = np.clip(goal-obs[16:18], -.1, .1); a[2]=np.clip(-obs[18],-.1,.1)
            obs, *_ = env.step(a)
            if np.linalg.norm(goal-obs[16:18])<0.005: break
    b = obs[54:57].copy()
    moved = False
    for t in range(200):
        a = np.zeros(11, np.float32); a[:2] = 0.01*d; a[2]=np.clip(-obs[18],-.1,.1)
        obs, *_ = env.step(a)
        if np.linalg.norm(obs[54:56]-b[:2])>0.002:
            print(direction, "contact: base", obs[16:19], "block", b[:2], "dist along", np.dot(b[:2]-obs[16:18], d)); moved=True; break
    if not moved: print(direction, "no contact", obs[16:18], b)
env.close()
