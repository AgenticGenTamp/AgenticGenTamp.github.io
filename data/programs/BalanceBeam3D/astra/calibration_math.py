import numpy as np
from scipy.optimize import least_squares
from kinematics_candidate import forward
HOME=np.array([0,-.3491,np.pi,-2.5482,0,-.8727,np.pi/2])
SEED=np.array([0,1.5,np.pi,-1.2,0,-.4,np.pi/2])
BOUNDS=([-6.28,-2.24,-6.28,-2.58,-6.28,-2.09,-6.28],[6.28,2.24,6.28,2.58,6.28,2.09,6.28])
def ik(x,z,q=SEED,mount=.4,extension=.12):
    def fun(q):
        t=forward(q,extension,(0,0,mount))
        return np.r_[5*(t[:3,3]-[x,0,z]),t[:3,2]-[0,0,-1], .0001*(q-SEED)]
    return least_squares(fun,q,bounds=BOUNDS,max_nfev=100).x
