import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach, cube_face_yaw
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info); ap._parse(obs)
for n,c in ap.cubes.items():
    yaw,tilt=cube_face_yaw(c["quat"])
    print(n, np.round(c["p"],3), "yaw%.2f"%yaw, ap._cube_color(n), [round(ap._axis_clearance(n, yaw+k*np.pi/2),3) for k in (0,1)])
