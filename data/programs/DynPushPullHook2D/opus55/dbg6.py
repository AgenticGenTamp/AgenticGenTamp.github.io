import sys,os,numpy as np
from env_client import make_env
from approach import GeneratedApproach, wrap
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed,options=({'object_count':int(os.environ['OC'])} if os.environ.get('OC') else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(upto):
    a=ap.get_action(obs); obs,r,term,tr,_=env.step(a)
d=ap._parse(obs); E,u,n=ap.hook_end(d); pre=E-0.55*u; th=d['hth']
obst=ap._obstacles(d, walls=True, hook=True, target=True)
names=['obs%d'%i for i in range(len(d['obs']))]+['target','s1','s2','floor','left','right']
cur=float(d['rth']); rx,ry=d['r']
for dd in [wrap(th-cur), wrap(th-cur)-2*np.pi*np.sign(wrap(th-cur))]:
    route=[(rx,ry,cur,d['arm']),(rx,min(ry,.5),cur,.24),(max(pre[0],.266),.5,cur+dd,.24),(max(pre[0],.266),pre[1],cur+dd,.24)]
    for i in range(3):
        bad=[names[j] for j,P in enumerate(obst) if not ap.path_free(route[i],route[i+1],[P],False,margin=0.02)]
        print(dd, i, bad)
for m in [0.01,0.0,-0.01,-0.03]:
    p0=(rx,ry,cur,d['arm']); print(m, ap.path_free(p0,p0,[obst[-5]],False,margin=m), ap.path_free(p0,(rx,.5,cur,.24),[obst[-5]],False,margin=m))
print(ap.robot_polys(rx,ry,cur,d['arm'],0.32,base=True)[0].round(2)); print(obst[-5].round(2))
