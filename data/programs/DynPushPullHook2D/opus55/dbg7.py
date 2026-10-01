import sys,os,numpy as np
from env_client import make_env
from approach import GeneratedApproach, wrap
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed,options=({'object_count':int(os.environ['OC'])} if os.environ.get('OC') else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(upto):
    a=ap.get_action(obs); obs,r,term,tr,_=env.step(a)
d=ap._parse(obs); E,u,n=ap.hook_end(d); pre=E-0.55*u; th=d['hth']
obst=ap._obstacles(d, walls=False, hook=True, target=True)
obst.append(np.array([[-1.0, 1.72], [4.5, 1.72], [4.5, 1.78], [-1.0, 1.78]]))
names=['obs%d'%i for i in range(len(d['obs']))]+['target','s1','s2','mid']
cur=float(d['rth']); rx,ry=d['r']; px=max(pre[0],.266); py=pre[1]
print('pre',pre, 'th',th,'cur',cur)
for dd in [wrap(th-cur), wrap(th-cur)-2*np.pi*np.sign(wrap(th-cur))]:
    route=[(rx,ry,cur,d['arm']),(rx,.5,cur,.24),(px,.5,cur,.24),(px,py,cur+dd,.24)]
    for i in range(3):
        bad=[names[j] for j,P in enumerate(obst) if not ap.path_free(route[i],route[i+1],[P],False,margin=0.02)]
        print(round(dd,2), i, bad)
print(ap._grasp_shortcut(d, pre, th))
for yv in (.38,.278):
  route=[(rx,ry,cur,d['arm']),(rx,yv,cur,.24),(px,yv,cur,.24),(px,py,th,.24)]
  for i in range(3):
    bad=[names[j] for j,P in enumerate(obst) if not ap.path_free(route[i],route[i+1],[P],False,margin=0.0 if i==0 else 0.02)]
    print(yv, i, bad)
