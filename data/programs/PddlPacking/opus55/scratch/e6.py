import numpy as np, sys
sys.argv=['x']
exec(open('scratch/e5.py').read().split('P=np.pi')[0])
from pr2fk import fk, P as PP, rotz, roty, rotx
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
def pts(base,q,p=PP):
    bx,by,bt=base; R=rotz(bt); t=np.array([bx,by,0.0]); out=[]
    t=t+R@np.array([p['sx'],p['sy'],p['sz']]); out.append(t.copy())
    R=R@rotz(q[0]); t=t+R@np.array([p['l_sh'],0,0]); out.append(t.copy())
    R=R@roty(q[1])@rotx(q[2]); t=t+R@np.array([p['l_up'],0,0]); out.append(t.copy())
    R=R@roty(q[3])@rotx(q[4]); t=t+R@np.array([p['l_fo'],0,0]); out.append(t.copy())
    R=R@roty(q[5])@rotx(q[6]); t=t+R@np.array([p['l_tool'],0,0]); out.append(t.copy())
    return np.array(out)
print("arm pts at base (0,0,0):\n",pts((0,0,0),q0).round(3))
P=np.pi
# alternate seeds for -x yaw0
for s in [1,5]:
    def feas2(path,final,q=None,s=s):
        o,_=env.reset(seed=s)
        for w in path: o,ok=goto(o,w,q); assert ok
        return goto(o,final,q,maxd=0.005)[1]
    lo,hi=-1.0,-0.3
    while hi-lo>0.0005:
        m=(lo+hi)/2
        if feas2([(-1,0,0)],(m,0,0)): lo=m
        else: hi=m
    print("seed",s,"-x yaw0",round(lo,4))
# arm swung: q1 rotated to point back/out to +y
qa=q0.copy(); qa[0]=2.0
print("arm qa pts:\n",pts((0,0,0),qa).round(3))
print("-x yaw0 armswung", round(bis([(-1,0,0)],lambda v:(v,0,0),-1,-0.3,q=qa),4))
print("-x yaw pi/4", round(bis([(-1,0,P/4)],lambda v:(v,0,P/4),-1,-0.3),4))
print("-x yaw -pi/4", round(bis([(-1,0,-P/4)],lambda v:(v,0,-P/4),-1,-0.3),4))
print("-x yaw -pi/2", round(bis([(-1,0,-P/2)],lambda v:(v,0,-P/2),-1,-0.3),4))
print("-x yaw0 at y=0.9 (beyond table side)", round(bis([(-1,0.9,0)],lambda v:(v,0.9,0),-1,0.9),4))
