"""Analytical Kinova Gen3 geometry and compact numerical inverse kinematics."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ORIGINS = np.array([[0,0,.15643], [0,.005375,-.12838],
                    [0,-.21038,-.006375], [0,.006375,-.21038],
                    [0,-.20843,-.006375], [0,.00017505,-.10593],
                    [0,-.10593,-.00017505]])
ANGLES = [np.pi,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2]
FIXED = np.array([Rotation.from_rotvec([a,0,0]).as_matrix() for a in ANGLES])
HOME = np.radians([0,-20,180,-146,0,-50,90])

def fk(q, gripper_length=.15):
    """Return grasp center xyz and orientation matrix relative to arm mount.

    Grasp center lies gripper_length beyond the Gen3 tool flange along +tool z.
    """
    rot=np.eye(3)
    pos=np.zeros(3)
    for i,angle in enumerate(q):
        pos=pos+rot@ORIGINS[i]
        c,s=np.cos(angle),np.sin(angle)
        rot=rot@FIXED[i]@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pos=pos+rot@np.array([0,0,-.061525-gripper_length])
    rot=rot@np.diag([1.,-1.,-1.])
    return pos,rot

def ik(position, orientation=None, q0=None, gripper_length=.15,
       max_nfev=100, lower=None, upper=None):
    """Solve a nearby arm configuration; orientation is a 3x3 tool matrix.

    The caller supplies mount-relative target; return q and position error.
    Optional joint limits are seven-vectors.
    """
    initial=np.asarray(HOME if q0 is None else q0,dtype=float)
    target=np.asarray(position,dtype=float)
    def residual(q):
        p,r=fk(q,gripper_length)
        errors=[p-target]
        if orientation is not None:
            errors.append(.25*Rotation.from_matrix(np.asarray(orientation)@r.T).as_rotvec())
        return np.concatenate(errors)
    bounds=(-np.inf,np.inf) if lower is None else (lower,upper)
    result=least_squares(residual, initial, bounds=bounds,max_nfev=max_nfev,
                         ftol=1e-7,xtol=1e-7,gtol=1e-7)
    return result.x,float(np.linalg.norm(fk(result.x,gripper_length)[0]-target))

def planar_ik(x,z,yaw=0.,pitch=0.,q0=None):
    """Solve the arm's well-conditioned sagittal configuration."""
    def build(v):
        q2,q4=v
        return np.array([0.,q2,np.pi,q4,0.,q2-q4-np.pi+pitch,yaw])
    def fun(v):
        q=build(v);p,r=fk(q)
        return [p[0]-x,p[2]-z, max(abs(q[5])-2.08,0)]
    start=[1.6,-1.6] if q0 is None else np.clip([q0[1],q0[3]],[-2.23,-2.56],[2.23,-.01])
    fit=least_squares(fun,start,bounds=([-2.23,-2.56],[2.23,-.01]),max_nfev=40)
    return build(fit.x),np.linalg.norm(fun(fit.x))
