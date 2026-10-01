import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach, cube_face_yaw
seed=int(sys.argv[1]); cnt=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed,options={"object_count":cnt})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info); ap._parse(obs)
for n,c in sorted(ap.cubes.items(), key=lambda kv: kv[1]["p"][2]):
    yaw,tilt=cube_face_yaw(c["quat"])
    print(n, np.round(c["p"],3), "yaw%.2f tilt%.2f"%(yaw,tilt), ap._cube_color(n)[:1], [round(ap._axis_clearance(n, yaw+k*np.pi/2),3) for k in (0,1)])
