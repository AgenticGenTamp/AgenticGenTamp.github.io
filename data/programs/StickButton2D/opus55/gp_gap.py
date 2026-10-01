import math,numpy as np
import gp_probe as P
from gp_probe import st,act,goto,rel
def corners(r):
    th=r['theta']; c,s=math.cos(th),math.sin(th); gx=r['x']+r['arm_joint']*c; gy=r['y']+r['arm_joint']*s
    return [(gx+c*u-s*v, gy+s*u+c*v) for u in(-.005,.005) for v in(-.035,.035)]
def gapf(r,sk,side):
    cs=corners(r)
    if side=='bot': return min(sk[1]-p[1] for p in cs)
    if side=='left': return min(sk[0]-p[0] for p in cs)
    return min(p[0]-(sk[0]+.05) for p in cs)
def trial(side,d,h,g,approach=True):
    P.obs,_=P.env.reset(seed=P.SEED); r,s=st(P.obs); sx,sy=s[0],s[1]
    if side=='bot': tx,ty,th=sx+.025,sy,math.pi/2+d
    elif side=='left': tx,ty,th=sx,sy+h,d
    else: tx,ty,th=sx+.05,sy+h,math.pi+d
    ux,uy=math.cos(th),math.sin(th); bx,by=tx-.3*ux,ty-.3*uy
    goto(r['x'],min(r['y'],sy-.4),th,.1); goto(bx,min(by,sy-.4),th,.1); goto(bx,by,th,.1); goto(bx,by,th,.2)
    r,s=st(P.obs); g0=gapf(r,s,side)
    # move along x/y so that gap becomes g (vac 0)
    need=g0-g; k=abs(ux) if side!='bot' else abs(uy)
    dist=need/k
    while dist>1e-9:
        stp=min(dist,.02); r,s=act(stp*ux,stp*uy,0,0,0.0); dist-=stp
    r,s=st(P.obs); gb=gapf(r,s,side)
    r1,s1=act(0,0,0,0,1.0)   # turn vacuum on without moving
    jump=(s1[0]-s[0],s1[1]-s[1],s1[2]-s[2])
    r0,s0=st(P.obs); rel0=rel(r0,s0)
    for _ in range(4): r,s=act(0,-0.03,0,0,1.0)
    mv=math.hypot(s[0]-s0[0],s[1]-s0[1]); dr=max(abs(a-b) for a,b in zip(rel(r,s),rel0))
    return 'gap %.4f'%gb, 'snap %.4f %.4f %.4f'%jump, 'followed' if mv>0.1 and dr<1e-3 else 'NOT (mv %.3f)'%mv
if __name__=="__main__":
 for side,d,h in (('bot',0,0),('bot',.6,0),('left',.3,.2),('right',-.6,.2)):
  for g in (0.0005,0.002,0.004,0.006,0.01,0.02):
   print(side,d,g,trial(side,d,h,g),flush=True)
