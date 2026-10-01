import pickle, numpy as np
from scipy.spatial.transform import Rotation as Rot
from fk import fk
data=pickle.load(open('cal_data2.pkl','rb'))+pickle.load(open('cal_data.pkl','rb'))
def pose(p):
    M=np.eye(4); M[:3,:3]=Rot.from_quat(p[3:7]).as_matrix(); M[:3,3]=p[:3]; return M
ep=[];er=[]
for s in data:
    M,_=fk(s['q'],base=s['base']); P=M@pose(s['gtf'])
    Pt=pose(s['part']); ep.append(np.linalg.norm(P[:3,3]-Pt[:3,3])); er.append(np.linalg.norm(Rot.from_matrix(P[:3,:3].T@Pt[:3,:3]).as_rotvec()))
print('pos err max',max(ep),'mean',np.mean(ep),'rot err max',max(er))
