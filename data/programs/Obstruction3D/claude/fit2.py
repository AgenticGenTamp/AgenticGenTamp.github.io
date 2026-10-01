import numpy as np, pickle
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
import fk
data=[]
for s in [0,1,2]:
    d=pickle.load(open(f'calib_{s}.pkl','rb'))
    for q,T,b in d['data']: data.append((q,T,b))
FIXED=[f.copy() for f in fk._FIXED]; EE=fk._EE.copy()
def fkp(q, offs, Xt):
    T=np.eye(4)
    for i in range(7):
        F=FIXED[i].copy(); F[:3,3]=F[:3,3]+offs[i]
        T = T @ F @ fk.rotz(q[i])
    T = T @ EE
    X=np.eye(4); X[:3,:3]=np.diag([1,-1,-1.]); X[:3,3]=Xt
    return T @ X
def resid(p):
    bz=p[0]; offs=p[1:22].reshape(7,3); Xt=p[22:25]
    out=[]
    for q,T,b in data:
        B=np.eye(4); c,s=np.cos(b[2]),np.sin(b[2]); B[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]; B[:3,3]=[b[0],b[1],bz]
        P = B @ fkp(q,offs,Xt)
        out.append(P[:3,3]-T[:3,3])
    return np.concatenate(out)
p0=np.zeros(25); p0[22:25]=[0.011,0.0098,0.1647]
sol=least_squares(resid,p0)
r=resid(sol.x).reshape(-1,3)
print("bz",sol.x[0])
print("offs",np.round(sol.x[1:22].reshape(7,3),5))
print("Xt",sol.x[22:25])
d=np.linalg.norm(r,axis=1); print("pos rms",d.mean(),"max",d.max())
np.save('fitparams.npy', sol.x)
