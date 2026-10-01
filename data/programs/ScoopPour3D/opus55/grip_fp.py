import sys, json
from grip_util import *
def best_spot(g, name):
    c=g.cubes(); others=[p for n,p in c.items() if n!=name]
    best=None
    for x in np.arange(0.34,0.62,0.01):
        for y in np.arange(-0.29,-0.11,0.01):
            d=min(np.linalg.norm(p[:2]-[x,y]) for p in others)
            if best is None or d>best[0]: best=(d,(x,y))
    return best
def get(g,name,spot):
    for dz in [-0.004,-0.002,-0.006,0.0]:
        ok,_=pick(g,name,0.6,dz,nclose=5)
        if ok: return place(g,name,spot)
tag=sys.argv[1]; trials=json.loads(sys.argv[2]); name='cube_14'
g=G(2); d,spot=best_spot(g,name); print('spot',spot,'clear',d,flush=True)
p=get(g,name,spot); print('placed',p,flush=True)
for t in trials:
    p=g.P(name)
    if np.linalg.norm(p[:2]-np.array(spot))>0.003 or abs(p[2]-0.4825)>0.0015:
        get(g,name,(p[0],p[1]) if False else spot)
        p=g.P(name); print('re',p,flush=True)
    ok,info=pick(g,name,**t)
    tr=dict(t); tr.update(ok=bool(ok), push=float(info['push']), rise=info['rise'].tolist(), cmove=info['cmove'].tolist(), tool_err=info['tool_err'].tolist(), held=info['held_off'].tolist(), pushz=float(info['pushz']))
    print(tag, json.dumps(tr), flush=True)
    if ok: place(g,name,spot)
