from probe_hook_lib import *
from probe_wall_9 import plan2
import numpy as np, sys
def grasp(p,a,arm,gap,depth,dy=0.005):
    h=p.h(); R,ti,xm=plan2(a,arm,gap,depth,xo=min(3.495,h['x']+0.015))
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+a,arm=arm,gap=gap)
    p.goto(x=R[0])
    while p.r()['y']>R[1]+1e-3:
        y0=p.r()['y']; p.step([0,max(-dy,R[1]-y0),0,0,0])
        if p.r()['y']==y0: break
    for k in range(20):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: return True
    return False
cfg=[float(v) for v in sys.argv[1].split(',')]
for seed in [1,8,9,14,3,0]:
    p=P(seed); h0=p.h(); ok=grasp(p,*cfg[:4],dy=cfg[4] if len(cfg)>4 else 0.005); r=p.r(); h=p.h()
    rel=(h['x']-r['x'],h['y']-r['y'],h['theta']-r['theta'])
    p.goto(y=r['y']+0.4); r2=p.r(); h2=p.h()
    rel2=(h2['x']-r2['x'],h2['y']-r2['y'],h2['theta']-r2['theta'])
    print(cfg,seed,'h0x %.3f'%h0['x'],'OK' if ok else 'FAIL','robot',r,'hook',h,'rel %.4f %.4f %.4f'%rel,'| lifted rel %.4f %.4f %.4f held %d'%(*rel2,h2['held']),flush=True); p.env.close()
