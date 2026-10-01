from probe_g44_2 import geo, goto2, descend, close
from probe_hook_lib import P
import numpy as np, sys
import os
A,ARM,GAP,DEPTH,DY=float(os.environ.get("A",0.3)),float(os.environ.get("ARM",0.4)),0.25,float(os.environ.get("DEPTH",0.08)),float(os.environ.get("DY",0.00125))
OFF=float(sys.argv[2]) if len(sys.argv)>2 else 0.015
def grasp(p):
    n0=0
    h=p.h(); xout=min(float(os.environ.get("XCLIP",3.499)),h["x"]+OFF)
    d,n,to,ti=geo(A,ARM,GAP); Rx=min(3.3,xout-to[0])
    c=0
    c+=abs(goto2(p,y=2.2)); c+=abs(goto2(p,x=3.0)); c+=abs(goto2(p,y=1.4))
    c+=abs(goto2(p,th=-np.pi/2+A,arm=ARM,gap=GAP)); c+=abs(goto2(p,x=Rx))
    y0=p.r()['y']; lg=[]; descend(p,0.5-DEPTH,dy=DY,log=lg,yfast=float(os.environ.get("YFAST",0.51))); c+=len(lg)
    pre=p.h(); k=close(p); c+=k if k else 25
    return k>0,c,pre,Rx
seed=int(sys.argv[1]); p=P(seed); h0=p.h()
ok,c,pre,Rx=grasp(p); r=p.r(); h=p.h()
def rel(r,h):
    dx,dy=h['x']-r['x'],h['y']-r['y']; t=r['theta']
    return 'dth %.4f  corner_in_robot_frame (%.4f, %.4f)  world_offset (%.4f, %.4f)'%(h['theta']-t, np.cos(t)*dx+np.sin(t)*dy, -np.sin(t)*dx+np.cos(t)*dy, dx,dy)
print(seed,'h0',h0,'Rx %.4f'%Rx,'OK' if ok else 'FAIL','steps',c,'pre',pre,'\n  r',r,'\n  h',h,'\n  ',rel(r,h))
y1=r['y']; m=goto2(p,y=y1+0.5); r2=p.r(); h2=p.h()
print('  lift steps',m,'r',r2,'h',h2,'\n  ',rel(r2,h2),flush=True)
