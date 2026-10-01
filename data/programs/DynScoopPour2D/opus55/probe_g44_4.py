from probe_g44_2 import *
import sys
def attempt(p,a,arm,gap,depth,xout,dy):
    d,n,to,ti=geo(a,arm,gap); Rx=min(3.3,xout-to[0])
    goto2(p,th=-np.pi/2+a,arm=arm,gap=gap); goto2(p,x=Rx)
    descend(p,0.5-depth,dy=dy)
    r=p.r(); h=p.h(); tix=r['x']+ti[0]
    return h['x']<tix-0.003, tix
def grasp(p,atts,dy=0.0025,depth=0.08,xout=3.499,arm=0.4,gap=0.25):
    goto2(p,y=2.2); goto2(p,x=3.0); goto2(p,y=1.4)
    hist=[]
    for a in atts:
        a,dya=(a if isinstance(a,tuple) else (a,dy))
        good,tix=attempt(p,a,arm,gap,depth,xout,dya); hist.append((a,p.h()['x'],round(tix,4),good))
        if good:
            k=close(p)
            if k: return True,hist
            goto2(p,gap=gap)
        # retreat up
        r=p.r(); goto2(p,y=r['y']+0.25)
    return False,hist
if __name__=='__main__':
    atts=[tuple(float(u) for u in v.split(':')) if ':' in v else float(v) for v in sys.argv[1].split(',')]; seed=int(sys.argv[2]); dy=float(sys.argv[3]) if len(sys.argv)>3 else 0.0025
    p=P(seed); h0=p.h(); ok,hist=grasp(p,atts,dy=dy); r=p.r(); h=p.h()
    print(sys.argv[1:],'h0x',h0['x'],'OK' if ok else 'FAIL',hist,'steps',p.env._steps if hasattr(p.env,'_steps') else '', 'r',r,'h',h,flush=True)
