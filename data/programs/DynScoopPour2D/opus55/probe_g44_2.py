from probe_hook_lib import *
import numpy as np, sys
def geo(a,arm,gap):
    th=-np.pi/2+a; d=np.array([np.cos(th),np.sin(th)]); n=np.array([-d[1],d[0]])
    return d,n,(arm+0.12)*d+(gap/2+0.02)*n,(arm+0.12)*d+(gap/2-0.02)*n
def goto2(p,x=None,y=None,th=None,arm=None,gap=None,maxn=200):
    last=None
    for i in range(maxn):
        r=p.r(); a=[0]*5
        if x is not None: a[0]=np.clip(x-r['x'],-.03,.03)
        if y is not None: a[1]=np.clip(y-r['y'],-.03,.03)
        if th is not None: a[2]=np.clip((th-r['theta']+np.pi)%(2*np.pi)-np.pi,-.098,.098)
        if arm is not None: a[3]=np.clip(arm-r['arm_joint'],-.08,.08)
        if gap is not None: a[4]=np.clip(gap-r['finger_gap'],-.015,.015)
        if max(abs(v) for v in a)<1e-3: return i
        p.step(a)
        if p.r()==r: return -i
    return maxn
def approach(p,a,arm,gap,xout=3.499):
    d,n,to,ti=geo(a,arm,gap)
    Rx=min(3.3,xout-to[0])
    goto2(p,y=2.2); goto2(p,x=3.0); goto2(p,y=1.4); goto2(p,th=-np.pi/2+a,arm=arm,gap=gap)
    goto2(p,x=Rx)
    return d,n,to,ti
def descend(p,ytip,dy=0.005,log=None,yfast=0.53):
    # lower until outer-finger inner tip reaches ytip
    while True:
        r=p.r(); _,_,to,ti=geo(r['theta']+np.pi/2,r['arm_joint'],r['finger_gap'])
        cur=r['y']+ti[1]
        if cur<=ytip+1e-4: return True
        p.step([0,max(-(0.03 if cur>yfast+0.03 else dy),ytip-cur),0,0,0])
        if log is not None: log.append((round(p.r()['y']+ti[1],4),p.h()['x'],p.h()['theta']))
        if p.r()==r: return False
def close(p,n=25,extra=None):
    for k in range(n):
        p.step(extra if extra is not None else [0,0,0,0,-0.015])
        if p.h()['held']==1: return k+1
    return 0
if __name__=='__main__':
    a,arm,gap,depth=[float(v) for v in sys.argv[1].split(',')]; seed=int(sys.argv[2]) if len(sys.argv)>2 else 44
    p=P(seed); h0=p.h(); approach(p,a,arm,gap)
    r=p.r(); _,_,to,ti=geo(a,arm,gap)
    log=[]; ok=descend(p,0.5-depth,log=log)
    pre=p.h(); k=close(p); r=p.r(); h=p.h()
    print(sys.argv[1],'h0x',h0['x'],'tipout_x %.4f tipin_x %.4f'%(r['x']+to[0],r['x']+ti[0]),'desc',ok,'pre',pre,'close',k,'r',r,'h',h,'log',log[::4][-8:],flush=True)
