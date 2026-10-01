import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach, quat_yaw
np.set_printoptions(precision=4, suppress=True, linewidth=200)
def tilt(o):
    w,x,y,z = o[41:45]
    # angle of rotated local x-axis (beam) out of horizontal
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    return np.degrees(np.arcsin(R[2,0]))
seed, obj, off = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
c = obs[38:41]; yaw = quat_yaw(obs[41:45]); axis = np.array([np.cos(yaw), np.sin(yaw), 0])
ap.tasks = [(obj, c + off*axis)]
print("seesaw", c, "yaw", yaw)
for t in range(400):
    obs, r, te, tr, info = env.step(ap.get_action(obs))
    if ap.task_i >= 1 and t % 5 == 0:
        print(t, "tilt %.2f" % tilt(obs), "obj", obs[obj:obj+3], "sw", obs[38:41], "w", obs[48:51], te)
    if t > 0 and ap.task_i >= 1 and t > ap_t0 + 60: break
    if ap.task_i < 1: ap_t0 = t
