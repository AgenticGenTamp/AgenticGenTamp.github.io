import numpy as np, json
d=json.load(open('cal_pts.json'))
P=[];U=[]
for p,uv,n in d:
    if uv is None: continue
    P.append(p); U.append(uv)
P=np.array(P); U=np.array(U)
A=[]
for (X,Y,Z),(u,v) in zip(P,U):
    A.append([X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z,-u])
    A.append([0,0,0,0,X,Y,Z,1,-v*X,-v*Y,-v*Z,-v])
A=np.array(A)
_,_,Vt=np.linalg.svd(A)
M=Vt[-1].reshape(3,4)
def proj(M,X):
    X=np.atleast_2d(X); h=np.c_[X,np.ones(len(X))]@M.T
    return h[:,:2]/h[:,2:3]
pred=proj(M,P)
err=np.linalg.norm(pred-U,axis=1)
print('n',len(P),'mean err px',round(err.mean(),2),'max',round(err.max(),2))
np.save('cam_M.npy',M)
print(np.round(M/M[2,3],4))
