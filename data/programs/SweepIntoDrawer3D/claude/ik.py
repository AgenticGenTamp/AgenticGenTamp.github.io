import numpy as np
from fk import fk, LINKS, TOOL, rotx, rotz, T
PI=np.pi
LIM_LO=np.array([-1e9,-2.41,-1e9,-2.66,-1e9,-2.23,-1e9])
LIM_HI=np.array([ 1e9, 2.41, 1e9, 2.66, 1e9, 2.23, 1e9])

def fk_all(q, tool_z=0.0):
    """returns list of transforms and EE transform"""
    M=np.eye(4); origins=[]; axes=[]
    for i,(xyz,rx) in enumerate(LINKS):
        M=M@T(rotx(rx),np.array(xyz))
        origins.append(M[:3,3].copy()); axes.append(M[:3,2].copy())
        M=M@T(rotz(q[i]),np.zeros(3))
    M=M@T(rotx(TOOL[1]),np.array(TOOL[0]))
    M=M@T(np.eye(3),np.array([0,0,tool_z]))
    return np.array(origins), np.array(axes), M

def jacobian(q, tool_z=0.0):
    o,a,M=fk_all(q,tool_z); p=M[:3,3]
    J=np.zeros((6,7))
    for i in range(7):
        J[:3,i]=np.cross(a[i], p-o[i])
        J[3:,i]=a[i]
    return J,M

def ik(target_p, target_R, q0, tool_z=0.0, iters=200, w_rot=1.0):
    q=np.array(q0,dtype=float)
    for k in range(iters):
        J,M=jacobian(q,tool_z)
        ep=target_p-M[:3,3]
        if target_R is None:
            e=np.concatenate([ep,[0,0,0]]); Ju=J.copy(); Ju[3:]=0
        else:
            Rerr=target_R@M[:3,:3].T
            ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1))
            if ang<1e-8: er=np.zeros(3)
            else:
                axis=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))
                er=axis*ang
            e=np.concatenate([ep, w_rot*er]); Ju=J
        if np.linalg.norm(e)<1e-5: break
        lam=0.05
        dq=Ju.T@np.linalg.solve(Ju@Ju.T+lam**2*np.eye(6), e)
        dq=np.clip(dq,-0.2,0.2)
        q=q+dq
        q=np.clip(q,LIM_LO,LIM_HI)
    J,M=jacobian(q,tool_z)
    err=np.linalg.norm(target_p-M[:3,3])
    return q, err
if __name__=="__main__":
    q0=np.array([0.0,-0.3491,3.1416,-2.5482,-0.0,-0.8727,1.5708])
    Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
    for tgt in [(0.5,0.0,0.2),(0.343,0.317,0.1),(0.6,0.3,0.0)]:
        q,e=ik(np.array(tgt),Rdown,q0)
        print(tgt,"err",round(e,5),"q",np.round(q,3))
