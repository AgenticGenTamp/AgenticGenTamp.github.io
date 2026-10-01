import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3, suppress=True, linewidth=200)
# seed 14: S2 (70) at (0.617,0.036), S1 at (0.66,0.024): neighbor along +x
for th_force in [0.0, np.pi/2]:
    env = make_env(); obs, info = env.reset(seed=14)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    ap.tasks = [(70, None)]
    orig = ap.best_pose
    ap.best_pose = lambda s, p, yaw, nsym=4, th0=th_force: np.array([p[0]-0.5*np.cos(th0), p[1]-0.5*np.sin(th0), th0])
    for t in range(150):
        obs, *_ = env.step(ap.get_action(obs))
        if ap.phase == "transport" and ap.t > 12: break
    print("heading", th_force, "S2", obs[70:73], "S1", obs[54:57])
    env.close()
