import numpy as np
from scipy.optimize import least_squares
from kinematics_candidate import forward
from calibration_math import HOME
bounds=([-6.28,-2.24,-6.28,-2.58,-6.28,-2.09,-6.28],[6.28,2.24,6.28,2.58,6.28,2.09,6.28])
for mount in [.3,.4,.5,.6,.7]:
 def f(q):
  t=forward(q,.12,(0,0,mount));return np.r_[5*(t[:3,3]-[.55,0,.025]),t[:3,2]-[0,0,-1],.0001*(q-HOME)]
 best=None
 for k in range(10):
  q=HOME if k==0 else np.random.default_rng(k).uniform(bounds[0],bounds[1]);sol=least_squares(f,q,bounds=bounds,max_nfev=120)
  if best is None or np.linalg.norm(sol.fun)<best[0]:best=(np.linalg.norm(sol.fun),sol.x)
 print(mount,best)
