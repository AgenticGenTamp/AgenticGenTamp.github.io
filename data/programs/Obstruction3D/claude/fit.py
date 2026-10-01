import numpy as np, pickle
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import fk
data=[]
for s in [0,1,2]:
    d=pickle.load(open(f'calib_{s}.pkl','rb'))
    for q,T,b in d['data']: data.append((q,T,b))
print(len(data))
def resid(p):
    bz=p[0]; X=np.eye(4); X[:3,3]=p[1:4]; X[:3,:3]=R.from_rotvec(p[4:7]).as_matrix()
    out=[]
    for q,T,b in data:
        B=np.eye(4); c,s=np.cos(b[2]),np.sin(b[2]); B[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]; B[:3,3]=[b[0],b[1],bz]
        P = B @ fk.fk_frames(q)[-1] @ X
        out.append(P[:3,3]-T[:3,3])
        Rerr = P[:3,:3] @ T[:3,:3].T
        out.append(R.from_matrix(Rerr).as_rotvec())
    return np.concatenate(out)
p0=np.zeros(7); p0[1:4]=[0.12,0,0.125]
sol=least_squares(resid,p0)
r=resid(sol.x).reshape(-1,3)
print("bz",sol.x[0],"Xt",sol.x[1:4],"Xrv",sol.x[4:7])
print("pos rms",np.sqrt((r[0::2]**2).sum(1)).mean(), "max",np.sqrt((r[0::2]**2).sum(1)).max())
print("rot rms",np.sqrt((r[1::2]**2).sum(1)).mean(), "max",np.sqrt((r[1::2]**2).sum(1)).max())
