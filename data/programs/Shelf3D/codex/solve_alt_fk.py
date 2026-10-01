"""Standard-DH Gen3 IK calibrated against the live rendered arm."""
import numpy as np
from scipy.optimize import least_squares

PI=np.pi
ALPHA=[PI/2]*6+[0.0]
D=[.2848,0.0,.4208,0.0,.3143,0.0,.1674]
OFF=[0.0,PI,PI,PI,PI,PI,PI]

def fk(q):
    out=np.eye(4)
    for qi,alpha,d,offset in zip(q,ALPHA,D,OFF):
        t=qi+offset; c=np.cos(t); s=np.sin(t); ca=np.cos(alpha); sa=np.sin(alpha)
        out=out@np.array([[c,-s*ca,s*sa,0.0],[s,c*ca,-c*sa,0.0],
                          [0.0,sa,ca,d],[0.0,0.0,0.0,1.0]])
    return out

def residual(q, z=-.27):
    t=fk(q)
    return np.r_[3*(t[:3,3]-[-.647,.078,z]), t[:3,2]-[0,0,-1]]

if __name__=='__main__':
    start=np.array([-.117,2.03,3.139,-.06,0,-1.043,1.571])
    lo=np.array([-PI,-2.24,-PI,-2.58,-PI,-2.1,-PI])
    hi=np.array([ PI, 2.24, PI, 2.58, PI, 2.1, PI])
    for z in (-.24,-.27,-.30,-.34):
        r=least_squares(lambda q:residual(q,z),start,bounds=(lo,hi),max_nfev=5000)
        t=fk(r.x); print(z,'cost',np.linalg.norm(r.fun),'q',np.round(r.x,4),
              'p',np.round(t[:3,3],4),'axis',np.round(t[:3,2],4))
