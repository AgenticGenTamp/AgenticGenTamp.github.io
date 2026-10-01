import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
sd=int(sys.argv[1]); N=int(sys.argv[2])
obs,info=env.reset(seed=sd)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(N):
    obs,r,term,tr,info=env.step(ap.get_action(obs))
    if term: break
print('term',term,i,info)
for n in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(n)
    if n=='robot': continue
    print(n, {f:round(float(obs.get(o,f)),4) for f in ['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw'] if True})
