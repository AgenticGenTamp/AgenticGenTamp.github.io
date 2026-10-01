import sys, json, itertools
from grip_util import *
HOME=(0.35,-0.28)
def acquire(g, name, home=HOME):
    for dz in [-0.012,-0.006,-0.017,0.0,-0.012,-0.006]:
        ok,info=pick(g,name,0.0,dz,nclose=8)
        print('acq',dz,ok,flush=True)
        if ok: return place(g,name,home)
    return None
def session(seed, name, trials, tag, home=HOME):
    g=G(seed); res=[]
    if home is None: home=tuple(g.P(name)[:2])
    else: p=acquire(g,name,home); print('home',p,flush=True)
    for t in trials:
        c=g.cubes(); 
        if not isolated(c,name,0.04) or abs(g.P(name)[2]-0.4825)>0.002:
            p=acquire(g,name,home); print('reacq',p,flush=True)
        ok,info=pick(g,name,**t)
        tr=dict(t); tr.update(ok=bool(ok), push=float(info['push']), rise=info['rise'].tolist(), cmove=info['cmove'].tolist(), tool_err=info['tool_err'].tolist(), held=info['held_off'].tolist())
        print(tag, json.dumps(tr), flush=True); res.append(tr)
        if ok: place(g,name,home)
    g.env.close()
    return res
if __name__=='__main__':
    tag=sys.argv[1]; seed=int(sys.argv[2]); name=sys.argv[3]; trials=json.loads(sys.argv[4])
    session(seed,name,trials,tag,None if len(sys.argv)<6 else HOME)
