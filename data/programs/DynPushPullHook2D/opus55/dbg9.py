import sys,os,numpy as np
from env_client import make_env
from approach import GeneratedApproach, wrap
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed,options=({'object_count':int(os.environ['OC'])} if os.environ.get('OC') else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
prev=None
for t in range(400):
    a=ap.get_action(obs)
    if prev=='sweep' and getattr(ap,'after',None)=='over': break
    prev=getattr(ap,'after',None)
    obs,r,term,tr,_=env.step(a)
d=ap._parse(obs)
print(t,'r',np.round(d['r'],3),d['rth'],d['arm'])
for p in ap.man: print(np.round(p,2))
wps,after=ap._plan(d)
poses=[np.r_[ap._robot_target(C,phi)[0], ap._robot_target(C,phi)[1:]] for C,phi in wps]
obst=ap._obstacles(d); cur=np.array((d['r'][0], d['r'][1], d['rth'], d['arm']))
pT=poses[1]
for yv in [0.39,0.51,0.63]:
    route=[np.array([cur[0],yv,cur[2],cur[3]]), np.array([pT[0],yv,pT[2],pT[3]]),pT]
    pv=cur; res=[]
    for i,w in enumerate(route):
        m=0.03
        res.append([j for j,P in enumerate(obst) if not ap.path_free(pv,w,[P],True,margin=m,start_ok=(i==0))]); pv=w
    print(yv,res)
