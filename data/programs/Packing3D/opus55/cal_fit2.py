import pickle, numpy as np
from fk import fk, Rz
data=pickle.load(open('cal_data2.pkl','rb'))+pickle.load(open('cal_data.pkl','rb'))
data=[s for s in data if s['ga']>0]
for mode in ['base_frame','world']:
    A=[];y=[]
    for s in data:
        M,_=fk(s['q'],base=s['base']); Rb=Rz(s['base'][2])
        A.append(np.hstack([Rb if mode=='base_frame' else np.eye(3),M[:3,:3]])); y.append(s['part'][:3]-M[:3,3])
    A=np.vstack(A); y=np.concatenate(y); x,*_=np.linalg.lstsq(A,y,rcond=None)
    res=(A@x-y).reshape(-1,3); print(mode,'off',x[:3],'tool',x[3:],'rms',np.sqrt((res**2).sum(1).mean()),'max',np.abs(res).max())
