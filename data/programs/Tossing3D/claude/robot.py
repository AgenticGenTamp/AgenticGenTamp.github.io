import numpy as np
PI=np.pi
def rpy(r,p,y):
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])
def H(xyz,r,p,y):
    M=np.eye(4); M[:3,:3]=rpy(r,p,y); M[:3,3]=xyz; return M
JOINTS=[((0,0,0.15643),(PI,0,0)),
        ((0,0.005375,-0.12838),(PI/2,0,0)),
        ((0,-0.21038,-0.006375),(-PI/2,0,0)),
        ((0,0.006375,-0.21038),(PI/2,0,0)),
        ((0,-0.20843,-0.006375),(-PI/2,0,0)),
        ((0,0.00017505,-0.10593),(PI/2,0,0)),
        ((0,-0.10593,-0.00017505),(-PI/2,0,0))]
EE=H((0,0,-0.0615),PI,0,0)
def Rz(t):
    M=np.eye(4); c,s=np.cos(t),np.sin(t); M[0,0]=c;M[0,1]=-s;M[1,0]=s;M[1,1]=c; return M
_PRE=[H(xyz,*r) for xyz,r in JOINTS]
def fk_all(q,tool=0.0):
    """Return list of frames; last is tool frame."""
    M=np.eye(4); frames=[]
    for i in range(7):
        M=M@_PRE[i]@Rz(q[i]); frames.append(M.copy())
    M=M@EE
    if tool: M=M@H((0,0,tool),0,0,0)
    frames.append(M.copy())
    return frames
def fk(q,tool=0.0):
    return fk_all(q,tool)[-1]
def jac(q,tool=0.0):
    fr=fk_all(q,tool); pe=fr[-1][:3,3]
    J=np.zeros((6,7))
    for i in range(7):
        z=fr[i][:3,2]; p=fr[i][:3,3]
        J[:3,i]=np.cross(z,pe-p); J[3:,i]=z
    return J
QLO=np.array([-1e9,-2.41,-1e9,-2.66,-1e9,-2.23,-1e9])
QHI=np.array([ 1e9, 2.41, 1e9, 2.66, 1e9, 2.23, 1e9])
def ik(target_pos, target_R, q0, tool=0.0, iters=120, w_rot=1.0):
    q=np.array(q0,dtype=float)
    for _ in range(iters):
        M=fk(q,tool); p=M[:3,3]; R=M[:3,:3]
        ep=target_pos-p
        if target_R is not None:
            Re=target_R@R.T
            ang=np.arccos(np.clip((np.trace(Re)-1)/2,-1,1))
            if ang<1e-8: er=np.zeros(3)
            else:
                er=ang/(2*np.sin(ang))*np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])
        else: er=np.zeros(3)
        e=np.concatenate([ep,w_rot*er])
        if np.linalg.norm(ep)<1e-4 and np.linalg.norm(er)<1e-3: break
        J=jac(q,tool)
        if target_R is None: J=J[:3]; e=ep
        lam=0.05
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(J.shape[0]), e)
        n=np.linalg.norm(dq)
        if n>0.2: dq*=0.2/n
        q=q+dq
        q=np.clip(q,QLO,QHI)
    return q

PLANAR_IDX=[1,3,5]
def planar_q(abc):
    q=np.zeros(7); q[2]=np.pi; q[6]=np.pi/2
    q[1],q[3],q[5]=abc
    return q
def ik_planar_multi(px,pz,pitch,ref,tool=0.0,n=7):
    """Grid-restart planar IK; returns solution closest to ref (in joint space)."""
    best=None
    lim=np.array([2.40,2.60,2.20])
    grid=np.linspace(-1,1,n)
    for ga in grid:
        for gb in grid:
            for gc in grid:
                abc=ik_planar(px,pz,pitch,(ga*2.3,gb*2.5,gc*2.1),tool=tool,iters=60)
                q=planar_q(abc); M=fk(q,tool); p=M[:3,3]; tz=M[:3,2]
                cur=np.arctan2(tz[2],tz[0])
                err=abs(px-p[0])+abs(pz-p[2])+abs((pitch-cur+np.pi)%(2*np.pi)-np.pi)
                if err>0.01: continue
                if np.any(np.abs(abc)>lim-0.02): continue
                d=float(np.sum(np.abs(abc-np.asarray(ref))))
                if best is None or d<best[0]: best=(d,abc)
    return None if best is None else best[1]

def ik_planar(px,pz,pitch,abc0=(0.0,0.0,0.0),tool=0.0,iters=200):
    """pitch: angle of tool z axis in x-z plane (atan2(tz_z,tz_x)); -pi/2 = pointing down"""
    abc=np.array(abc0,dtype=float)
    for _ in range(iters):
        q=planar_q(abc); M=fk(q,tool)
        p=M[:3,3]; tz=M[:3,2]
        cur=np.arctan2(tz[2],tz[0])
        e=np.array([px-p[0], pz-p[2], (pitch-cur+np.pi)%(2*np.pi)-np.pi])
        if np.max(np.abs(e))<1e-4: break
        Jf=jac(q,tool)
        Jp=np.array([Jf[0,PLANAR_IDX], Jf[2,PLANAR_IDX], Jf[4,PLANAR_IDX]])
        lam=0.03
        d=Jp.T@np.linalg.solve(Jp@Jp.T+lam**2*np.eye(3), e)
        n=np.linalg.norm(d)
        if n>0.25: d*=0.25/n
        abc=abc+d
        abc=np.clip(abc,[-2.40,-2.60,-2.20],[2.40,2.60,2.20])
    return abc
