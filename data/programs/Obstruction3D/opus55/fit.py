import numpy as np, json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from kin import *
D=json.load(open('calib_data.json'))
def pose(v):
    M=np.eye(4); M[:3,:3]=Rot.from_quat(v[3:7]).as_matrix(); M[:3,3]=v[:3]; return M
g=np.array([d['gtf'] for d in D]); print('gtf std', g.std(0).round(5), g[0].round(4))
EEt=[pose(d['blk'])@np.linalg.inv(pose(d['gtf'])) for d in D]
def chain(base,q):
    return fk(base,q)
def Tp(p):
    M=np.eye(4); M[:3,:3]=Rot.from_rotvec(p[3:6]).as_matrix(); M[:3,3]=p[:3]; return M
def res(p):
    A=Tp(p[:6]); B=Tp(p[6:12]); r=[]
    for d,E in zip(D,EEt):
        bx,by,br=d['base']
        Bw=Rz(br); Bw[0,3]=bx; Bw[1,3]=by
        M=Bw@A@np.linalg.inv(Bw)@fk(d['base'],d['q'])@B  # A applied in base frame
        r.extend(M[:3,3]-E[:3,3]); r.extend(0.1*(M[:3,:3]-E[:3,:3]).ravel())
    return np.array(r)
sol=least_squares(res,np.zeros(12))
np.set_printoptions(precision=5,suppress=True)
print('A',sol.x[:6]); print('B',sol.x[6:]); r=res(sol.x).reshape(len(D),-1)
print('pos resid max',np.abs(r[:,:3]).max(),'rot resid max',np.abs(r[:,3:]).max())
print('raw EE pos err (no fit) first',EEt[0][:3,3]-fk(D[0]['base'],D[0]['q'])[:3,3])
print(EEt[0].round(4)); print(fk(D[0]['base'],D[0]['q']).round(4))
