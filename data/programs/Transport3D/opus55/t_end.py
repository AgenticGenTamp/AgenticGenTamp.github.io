import sys, numpy as np, approach
from env_client import make_env
env=make_env(); ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=1); ap.reset(obs,info)
for t in range(400):
    obs,r,term,trunc,info=env.step(ap.get_action(obs))
    if term: print('TERM',t); break
F=['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw','grasp_active','half_extent_z']
for n in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(n)
    if n!='robot': print(n,[round(float(obs.get(o,f)),5) for f in F])
r=obs.get_object_from_name('robot'); print('robot',[round(float(obs.get(r,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot','grasp_active']])
print(info)
