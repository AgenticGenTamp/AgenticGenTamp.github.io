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
P=np.pi
cases={
 "-x yaw0": ([(-1,0,0)], lambda v:(v,0,0), -1.0,-0.3),
 "-x yawpi": ([(-1,0,P)], lambda v:(v,0,P), -1.0,-0.3),
 "-x yaw pi/2": ([(-1,0,P/2)], lambda v:(v,0,P/2), -1.0,-0.3),
 "+x yaw0": ([(-1,-1.5,0),(1,-1.5,0),(1,0,0)], lambda v:(v,0,0), 1.0,0.3),
 "+x yawpi": ([(-1,-1.5,P),(1,-1.5,P),(1,0,P)], lambda v:(v,0,P), 1.0,0.3),
 "-y yaw0": ([(-1,-1.5,0),(0,-1.5,0)], lambda v:(0,v,0), -1.5,-0.6),
 "-y yawpi/2": ([(-1,-1.5,P/2),(0,-1.5,P/2)], lambda v:(0,v,P/2), -1.5,-0.6),
 "-y yaw-pi/2": ([(-1,-1.5,-P/2),(0,-1.5,-P/2)], lambda v:(0,v,-P/2), -1.5,-0.6),
 "+y yaw0": ([(-1,1.5,0),(0,1.5,0)], lambda v:(0,v,0), 1.5,0.6),
 "+y yawpi/2": ([(-1,1.5,P/2),(0,1.5,P/2)], lambda v:(0,v,P/2), 1.5,0.6),
 "+y yaw-pi/2": ([(-1,1.5,-P/2),(0,1.5,-P/2)], lambda v:(0,v,-P/2), 1.5,0.6),
}
for k,(p,mk,lo,hi) in cases.items():
    v=bis(p,mk,lo,hi); edge=0.3 if 'x' in k.split()[0] else 0.6
    print(k, "limit", round(v,4), "gap to edge", round(abs(v)-edge,4))
