import numpy as np, json, sys
from ik import fk_full
def load(fn):
    d=json.load(open(fn)); base=np.array(d['base']); X=[];Y=[]
    for r in d['rows']:
        q=np.array(r['q']); M,Ms=fk_full(q)
        Rs=[np.eye(3)]+[m[:3,:3] for m in Ms]   # R_0..R_6 (=Ms[0..5]) and R_7=Ms[6]
        A=np.hstack(Rs)   # 3 x 24  (8 blocks: I, Ms0..Ms6)
        X.append(A); Y.append(np.array(r['part'][:3])-np.array([base[0],base[1],0.0]))
    return np.vstack(X), np.concatenate(Y), d
X,Y,d=load('calib.json')
n=len(d['rows'])
# fit on first 18, test on rest
tr=slice(0,18*3); te=slice(18*3,n*3)
a,*_=np.linalg.lstsq(X[tr],Y[tr],rcond=None)
pred=X@a
err=(pred-Y).reshape(-1,3)
print("params",np.round(a.reshape(8,3),5))
print("train rms",np.round(np.linalg.norm(err[:18],axis=1).mean(),5),"max",np.round(np.linalg.norm(err[:18],axis=1).max(),5))
print("test  rms",np.round(np.linalg.norm(err[18:],axis=1).mean(),5),"max",np.round(np.linalg.norm(err[18:],axis=1).max(),5))
a2,*_=np.linalg.lstsq(X,Y,rcond=None)
e2=(X@a2-Y).reshape(-1,3); print("full fit rms",np.round(np.linalg.norm(e2,axis=1).mean(),5),"max",np.round(np.linalg.norm(e2,axis=1).max(),5))
np.save('lin_params.npy',a2.reshape(8,3))
