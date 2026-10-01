from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
f=env.observation_space.type_features[T("robot")]
def rs(o):
    r=o.get_objects(T("robot"))[0]; return np.array([o.get(r,x) for x in f])
def goto(o,tgt,q=None,maxd=0.2):
    for _ in range(200):
        s=rs(o); d=np.zeros(11,np.float32)
        d[:3]=np.array(tgt)-s[:3]; d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        if q is not None: d[3:10]=np.array(q)-s[3:10]
        if np.abs(d[:10]).max()<1e-6: return o,True
        d[:10]=np.clip(d[:10],-maxd,maxd); o2,*_=env.step(d)
        if np.allclose(rs(o2)[:10],s[:10]): return o2,False
        o=o2
    return o,False
def feas(path, final, q=None):
    o,_=env.reset(seed=0)
    for w in path:
        o,ok=goto(o,w,q)
        assert ok,("path fail",w)
    o,ok=goto(o,final,q,maxd=0.005)
    return ok
def bis(path, mk, lo, hi, q=None):  # lo feasible, hi infeasible
    assert feas(path,mk(lo),q) and not feas(path,mk(hi),q)
    while abs(hi-lo)>0.0005:
        m=(lo+hi)/2
        if feas(path,mk(m),q): lo=m
        else: hi=m
    return lo
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
