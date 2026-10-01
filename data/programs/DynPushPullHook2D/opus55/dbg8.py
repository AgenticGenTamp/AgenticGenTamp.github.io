import sys,os,numpy as np
from env_client import make_env
from approach import GeneratedApproach, wrap
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed,options=({'object_count':int(os.environ['OC'])} if os.environ.get('OC') else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(400):
    a=ap.get_action(obs)
    if ap.phase=='man': break
    obs,r,term,tr,_=env.step(a)
d=ap._parse(obs)
print('r',d['r'],d['rth'],d['arm'])
for p in ap.man: print(np.round(p,2))
wps,after=ap._plan(d); print(after,[ (np.round(C,2),round(phi,2)) for C,phi in wps])
poses=[np.r_[ap._robot_target(C,phi)[0], ap._robot_target(C,phi)[1:]] for C,phi in wps]
print([np.round(p,2) for p in poses])
obst=ap._obstacles(d)
cur=np.array((d['r'][0], d['r'][1], d['rth'], d['arm']))
for yv in [0.268,0.5,0.8]:
    route=[np.array([cur[0],yv,cur[2],cur[3]]), np.array([poses[0][0],yv,cur[2],cur[3]]), np.array([poses[0][0],yv,poses[0][2],poses[0][3]])]+poses[1:]
    prev=cur
    for i,w in enumerate(route):
        w=w.copy(); w[2]=prev[2]+wrap(w[2]-prev[2])
        print(yv,i,[j for j,P in enumerate(obst) if not ap.path_free(prev,w,[P],True)], len(obst))
        prev=w
for P in obst: print('ob',P.min(0).round(2),P.max(0).round(2))
for S in ap.pose_shapes(poses[0],True): print('sh',S.min(0).round(2),S.max(0).round(2))
