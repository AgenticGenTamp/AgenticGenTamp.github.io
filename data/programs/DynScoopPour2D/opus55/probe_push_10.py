from probe_push_9 import press2
from probe_push_lib import *
import numpy as np, sys
def grasp(p, clip=3.495, dy=0.005, yfast=0.1, xoff=0.0):
    h=p.h(); x=min(3.3,min(clip,h['x']+0.015)-0.20568)+xoff
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+0.2,arm=0.2,gap=0.25); p.goto(x=x)
    yt=h['y']+0.71376; n=0
    while p.r()['y']>yt+1e-4:
        y0=p.r()['y']; p.step([0,max(-(0.03 if y0>yt+yfast else dy),yt-y0),0,0,0]); n+=1
        if p.r()['y']==y0: break
    for k in range(1,31):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: break
    ok=p.h()['held']==1; hy=p.h()['y']; y1=p.r()['y']; p.goto(y=y1+0.3)
    return ok and p.h()['y']-hy>0.25
if __name__=="__main__":
  mode=sys.argv[1]; seed=int(sys.argv[2])
  res=[]
  for xo in [0,-0.0003,0.0003,-0.0006,0.0006,0.001]:
    for yf in [0.1,0.2]:
      p=P(seed); h0=p.h()
      if mode=='press':
          press2(p,0.2,0.2,3.499+xo,0.015,0.12); p.goto(y=1.2); hm=p.h()['x']; ok=grasp(p,yfast=yf)
      else:
          hm=h0['x']; ok=grasp(p,clip=float(mode),xoff=xo,yfast=yf)
      res.append((ok,round(hm,3))); p.env.close()
  print(mode,seed,'ok %d/%d'%(sum(r[0] for r in res),len(res)),res,flush=True)
  