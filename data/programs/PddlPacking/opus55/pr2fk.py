import numpy as np

def rotx(a):
    c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def roty(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rotz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# params: shoulder offset in base frame (x,y,z), link lengths
P = dict(sx=-0.05, sy=0.188, sz=0.790675+0.2, l_sh=0.1, l_up=0.4, l_fo=0.321, l_tool=0.18)

def fk(base, q, p=P):
    """base=(x,y,yaw), q=7 joints. returns tool pos (3,), rot (3,3)."""
    bx,by,bt=base
    R=rotz(bt); t=np.array([bx,by,0.0])
    t=t+R@np.array([p['sx'],p['sy'],p['sz']])
    R=R@rotz(q[0])
    t=t+R@np.array([p['l_sh'],0,0])
    R=R@roty(q[1])
    R=R@rotx(q[2])
    t=t+R@np.array([p['l_up'],0,0])
    R=R@roty(q[3])
    R=R@rotx(q[4])
    t=t+R@np.array([p['l_fo'],0,0])
    R=R@roty(q[5])
    R=R@rotx(q[6])
    t=t+R@np.array([p['l_tool'],0,0])
    return t,R

def quat_to_R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

from scipy.optimize import least_squares
LO=np.array([-0.7146,-0.5236,-0.8,-2.3213,-np.pi,-2.0943,-np.pi])
HI=np.array([2.2854,1.3963,3.9,0.0,np.pi,0.0,np.pi])

def ik(base, pos, Rdes, q_init, p=P, wrot=0.3):
    def res(q):
        t,R=fk(base,q,p)
        return np.concatenate([t-pos, wrot*(R-Rdes).ravel()])
    sol=least_squares(res, np.clip(q_init,LO+1e-6,HI-1e-6), bounds=(LO,HI))
    return sol.x, np.linalg.norm(sol.fun)

def R_to_quat(R):
    tr=np.trace(R)
    if tr>0:
        s=np.sqrt(tr+1.0)*2; w=0.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return np.array([x,y,z,w])
