import numpy as np
from math import cos, sin, pi
def rpy(r,p,y):
    cr,sr,cp,sp,cy,sy=cos(r),sin(r),cos(p),sin(p),cos(y),sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],[sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[-sp, cp*sr, cp*cr]])
def T(xyz, R):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=xyz; return M
def rz(q): return rpy(0,0,q)
JOINTS=[((0,0,0.15643),(pi,0,0)),((0,0.005375,-0.12838),(pi/2,0,0)),((0,-0.21038,-0.006375),(-pi/2,0,0)),
        ((0,0.006375,-0.21038),(pi/2,0,0)),((0,-0.20843,-0.006375),(-pi/2,0,0)),((0,0.00017505,-0.10593),(pi/2,0,0)),
        ((0,-0.10593,-0.00017505),(-pi/2,0,0))]
JT=[T(x,rpy(*r)) for x,r in JOINTS]
def fk_arm(q, tool=0.0):
    M=np.eye(4)
    for i in range(7):
        M=M@JT[i]@T((0,0,0),rz(q[i]))
    M=M@T((0,0,-0.061525),rpy(pi,0,0))@T((0,0,tool),np.eye(3))
    return M
if __name__=="__main__":
    q=[0,-0.35,-3.142,-2.5,0,-0.87,1.571]
    np.set_printoptions(precision=3,suppress=True)
    print(fk_arm(q)); print(fk_arm([0]*7))

def rot_err(Rc, Rt):
    E = Rt @ Rc.T
    return 0.5*np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])

def ik(target_p, target_R, q0, tool=0.0, iters=200, wrot=0.5):
    q=np.array(q0,dtype=float)
    for it in range(iters):
        M=fk_arm(q,tool)
        ep=target_p-M[:3,3]; er=rot_err(M[:3,:3],target_R)
        e=np.concatenate([ep,wrot*er])
        if np.linalg.norm(ep)<1e-4 and np.linalg.norm(er)<1e-3: break
        Jm=np.zeros((6,7)); d=1e-5
        for i in range(7):
            qd=q.copy(); qd[i]+=d; Md=fk_arm(qd,tool)
            Jm[:3,i]=(Md[:3,3]-M[:3,3])/d
            Jm[3:,i]=wrot*rot_err(M[:3,:3],Md[:3,:3])/d
        lam=0.01
        dq=Jm.T@np.linalg.solve(Jm@Jm.T+lam*np.eye(6),e)
        n=np.linalg.norm(dq)
        if n>0.3: dq*=0.3/n
        q+=dq
    return q, np.linalg.norm(ep), np.linalg.norm(er)
