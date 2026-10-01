from env_client import make_env
import numpy as np, sys
env = make_env()
def goto(obs, goal, th=0.0, n=100):
    for t in range(n):
        a = np.zeros(11, np.float32); a[:2] = np.clip(goal-obs[16:18], -.1, .1); a[2]=np.clip(th-obs[18],-.1,.1)
        obs, *_ = env.step(a)
        if np.linalg.norm(goal-obs[16:18])<0.003 and abs(th-obs[18])<0.01: break
    return obs
for direction, th in [("x",0.0), ("x",np.pi/2), ("y", 0.0), ("-y", 0.0), ("x", np.pi/4)]:
    obs, _ = env.reset(seed=0)
    q0 = obs[41:45].copy(); c = obs[38:41].copy()  # seesaw center, beam along y, half len 0.36
    if direction == "x": start = np.array([c[0]-1.0, c[1]]); d = np.array([1,0])
    elif direction == "y": start = np.array([c[0], c[1]-1.2]); d = np.array([0,1])
    else: start = np.array([c[0], c[1]+1.2]); d = np.array([0,-1])
    if direction != "x":
        obs = goto(obs, np.array([obs[16], start[1]]), th)
    obs = goto(obs, start, th)
    for t in range(300):
        a = np.zeros(11, np.float32); a[:2] = 0.01*d; a[2]=np.clip(th-obs[18],-.1,.1)
        obs, *_ = env.step(a)
        if np.linalg.norm(obs[38:40]-c[:2])>0.002 or abs(obs[44]-q0[3])>0.003:
            print(direction, "th", th, "contact: base", obs[16:19], "seesaw c", c[:2], "dist along", np.dot(c[:2]-obs[16:18], d)); break
env.close()
