"""Offline approximate Gen3 URDF kinematics pose search."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

ORIGINS = [
    ([0, 0, .15643], [np.pi, 0, 0]),
    ([0, .005375, -.12838], [np.pi/2, 0, np.pi]),
    ([0, -.21038, -.006375], [-np.pi/2, 0, np.pi]),
    ([0, .006375, -.21038], [np.pi/2, 0, np.pi]),
    ([0, -.20843, -.006375], [-np.pi/2, 0, np.pi]),
    # The Gen3 URDF joint-6 origin is on the negative local-z side.  The
    # positive sign sometimes quoted for DH tables is in a different frame.
    ([0, 0, -.10593], [np.pi/2, 0, np.pi]),
    ([0, -.10593, 0], [-np.pi/2, 0, np.pi]),
]

def tr(x, r):
    t = np.eye(4); t[:3,:3] = Rotation.from_euler('xyz', r).as_matrix(); t[:3,3] = x
    return t

def fk(q):
    t = np.eye(4)
    for qi, (x, r) in zip(q, ORIGINS): t = t @ tr(x, r) @ tr([0,0,0], [0,0,qi])
    return t @ tr([0,0,-.0615], [0,0,0])

def cost(q):
    t=fk(q); p=t[:3,3]; axis=t[:3,2]
    return (p[0]-.40)**2 + p[1]**2 + (p[2]+.38)**2 + .08*np.sum((axis-[0,0,-1])**2)

bounds=[(-np.pi,np.pi),(-2.24,2.24),(-np.pi,np.pi),(-2.58,2.58),(-np.pi,np.pi),(-2.1,2.1),(-np.pi,np.pi)]
if __name__ == "__main__":
    for seed in range(3):
        r=differential_evolution(cost,bounds,seed=seed,popsize=12,maxiter=300,tol=1e-8)
        t=fk(r.x); print(r.fun,np.round(r.x,3),np.round(t[:3,3],3),np.round(t[:3,2],3))
