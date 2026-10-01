from probe_push_lib import *
import numpy as np, sys
def fpts(r):
    th=r['theta']; d=np.array([np.cos(th),np.sin(th)]); n=np.array([-d[1],d[0]]); a=r['arm_joint']; g=r['finger_gap']
    base=np.array([r['x'],r['y']])
    # outer finger (+n side for theta near -pi/2 => right)
    pts=[base+(a+l)*d+(s*(g/2+w))*n for l in (0,0.12) for w in (-0.02,0.02) for s in (1,-1)]
    return np.array(pts)
def press(p,A,ARM,GAP,XR,DY,K,y0=0.52):
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+A,arm=ARM,gap=GAP)
    r=p.r(); P_=fpts(r); dx=XR-P_[:,0].max(); p.goto(x=r['x']+dx)
    r=p.r(); P_=fpts(r)
    # lowest point to y0
    p.goto(y=r['y']+(y0-P_[:,1].min()))
    tr=[]
    for i in range(K):
        yb=p.r()['y']; p.step([0,-DY,0,0,0]); tr.append((round(fpts(p.r())[:,1].min(),3),p.h()['x']))
        if p.r()['y']==yb: tr.append('blk'); break
    return tr
if __name__=='__main__':
    seed=int(sys.argv[1]); A,ARM,GAP,XR,DY=[float(v) for v in sys.argv[2:7]]; K=int(sys.argv[7])
    p=P(seed); h0=p.h(); tr=press(p,A,ARM,GAP,XR,DY,K); r=p.r()
    p.goto(y=1.2); h1=p.h()
    print(sys.argv[1:],'Rx %.3f'%r['x'],'moved %.4f'%(h0['x']-h1['x']),'h1',h1,'tr',tr[:12],flush=True)
