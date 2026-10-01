import numpy as np, kin
MOUNT=np.array([0.1197,0.0003,0.3952]); TOOL=0.1009
LIM=np.array([[-1e6,-2.234,-1e6,-2.58,-1e6,-2.109,-1e6],[1e6,2.234,1e6,2.58,1e6,2.109,1e6]])
def rq(o):
    r=o.get_object_from_name('robot'); f=o.type_features[r.type]; d=o.data[r]
    return np.array([d[f.index('pos_arm_joint%d'%(i+1))] for i in range(7)]), np.array([d[0],d[1],d[2]])
def rotz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1.0]])
def ik_multi(T,q0,n=20,tool=TOOL,rng=None):
    rng=rng or np.random.default_rng(0); best=None
    for k in range(n):
        s=q0 if k==0 else np.clip(q0+rng.uniform(-2,2,7),LIM[0],LIM[1])
        qd,ep,er=kin.ik(T,s,tool_z=tool,q_lim=LIM)
        if ep>0.004 or er>0.05: continue
        sc=np.max(np.abs(qd-q0))
        if best is None or sc<best[0]: best=(sc,qd)
    return None if best is None else best[1]
def world_to_arm(pos,R,b):
    Rb=rotz(b[2])
    T=np.eye(4); T[:3,:3]=Rb.T@R; T[:3,3]=Rb.T@(np.asarray(pos)-np.array([b[0],b[1],0]))-MOUNT
    return T
