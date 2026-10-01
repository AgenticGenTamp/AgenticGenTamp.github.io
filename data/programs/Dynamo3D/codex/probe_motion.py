import sys
import numpy as np
from env_client import make_env

seed = int(sys.argv[1]); axis = int(sys.argv[2]); val = float(sys.argv[3]); steps = int(sys.argv[4])
env = make_env(); s, info = env.reset(seed=seed)
rob = s.get_object_from_name("robot")
def pose(st): return tuple(float(st.get(rob, f)) for f in ("pos_base_x", "pos_base_y", "pos_base_rot"))
print("start", info, pose(s), flush=True)
a = np.zeros(11, np.float32); a[axis] = val
for i in range(steps):
    s, r, t, tr, inf = env.step(a)
    if i in (0, 4, 9, 19, steps-1) or t or tr:
        chairs=[]
        for o in s.get_objects(env.observation_space.get_type("mujoco_movable_object")):
            chairs.append((o.name, round(float(s.get(o,"x")),3),round(float(s.get(o,"y")),3)))
        print(i+1, "r",r,"done",t,tr,"pose",pose(s),"chairs",chairs, flush=True)
    if t or tr: break
env.close()
