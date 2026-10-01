import pickle, numpy as np
from scipy.spatial.transform import Rotation as Rot
from fk import fk
data=pickle.load(open('cal_data.pkl','rb'))
def pose(p):
    M=np.eye(4); M[:3,:3]=Rot.from_quat(p[3:7]).as_matrix(); M[:3,3]=p[:3]; return M
rel=[]
for s in data:
    M,_=fk(s['q'],base=s['base']); P=pose(s['part'])
    X=np.linalg.inv(M)@P; rel.append(np.r_[X[:3,3],Rot.from_matrix(X[:3,:3]).as_rotvec()])
rel=np.array(rel); print('mean',np.round(rel.mean(0),4)); print('std',np.round(rel.std(0),5))
print('gtf',np.round(data[0]['gtf'],4))
A=[];y=[]
for s in data:
    M,_=fk(s['q'],base=s['base']); A.append(np.hstack([np.eye(3),M[:3,:3]])); y.append(s['part'][:3]-M[:3,3])
A=np.vstack(A); y=np.concatenate(y); x,*_=np.linalg.lstsq(A,y,rcond=None)
res=(A@x-y).reshape(-1,3); print('b_off',np.round(x[:3],4),'tool',np.round(x[3:],4)); print('rms',np.sqrt((res**2).sum(1).mean()),'max',np.abs(res).max())
