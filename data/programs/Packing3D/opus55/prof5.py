import numpy as np, ik10
c0=np.array([-0.12,0,0, 0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2])
tp=np.array([0.3,0.0,0.2])
import scipy.optimize as so
orig_ls=so.least_squares; orig_min=so.minimize
def ls(*a,**k):
    r=orig_ls(*a,**k); print('LS', np.round(r.x,3)); return r
def mn(*a,**k):
    r=orig_min(*a,**k); print('MIN', r.status, np.round(r.x,3)); return r
ik10.least_squares=ls; ik10.minimize=mn
c,md,err=ik10.solve(c0,tp,None,yaw_free=True,use_base=False)
print(np.round(c-c0,3))
