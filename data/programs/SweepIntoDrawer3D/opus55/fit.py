import numpy as np, kin
from scipy.optimize import least_squares
D=np.load('calib_data.npy')[:6]
def res(p):
    kin.MOUNT=p[:3]; r=[]
    for d in D:
        T=kin.fk_world(d[:3],d[3:10]); r.append(T[:3,:3]@p[3:6]+T[:3,3]-d[10:13])
    return np.concatenate(r)
s=least_squares(res,np.r_[0.12,0,0.4,0,0,0]); print(s.x, np.abs(s.fun).max())
