import sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for sd in map(int,sys.argv[1:]):
    obs,info=env.reset(seed=sd)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    names=[n for n in ap.objs if n!='robot']
    b=ap._robot()[0]
    print(sd,'robot',b.round(2))
    for n in names:
        o=ap._obj(n); x0,x1,y0,y1,_=ap.table
        tc=max(x0-o['x'],o['x']-x1,y0-o['y'],o['y']-y1)
        cl={m:round(ap._clear(ap._obj(m),[o['x'],o['y']])-max(o['hx'],o['hy']),3) for m in names if m!=n}
        print('  ',n,round(o['x'],3),round(o['y'],3),'yaw',round(__import__('approach').quat_yaw(*o['q']),2),'tabledist',round(tc,3),cl)
