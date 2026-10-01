import numpy as np
from env_client import make_env
C=[];W=[];B=[];Q=[]
for s in range(30):
    env=make_env(); obs,_=env.reset(seed=s)
    C.append(obs[:80].reshape(5,16)[:,:3]); W.append(obs[147:154]); B.append(obs[125:128]); Q.append(obs[128:136])
    env.close()
C=np.array(C); W=np.array(W); B=np.array(B); Q=np.array(Q)
f=C.reshape(-1,3)
print("cube x min/max/mean %.3f %.3f %.3f"%(f[:,0].min(),f[:,0].max(),f[:,0].mean()))
print("cube y min/max/mean %.3f %.3f %.3f"%(f[:,1].min(),f[:,1].max(),f[:,1].mean()))
print("cube z uniq",np.unique(np.round(f[:,2],3)))
print("wiper x %.3f..%.3f y %.3f..%.3f z"%(W[:,0].min(),W[:,0].max(),W[:,1].min(),W[:,1].max()),np.unique(np.round(W[:,2],3)),"quat0",np.round(W[0,3:7],3))
print("base x %.3f..%.3f y %.3f..%.3f yaw %.3f..%.3f"%(B[:,0].min(),B[:,0].max(),B[:,1].min(),B[:,1].max(),B[:,2].min(),B[:,2].max()))
print("q0 same across seeds:",np.allclose(Q,Q[0]),np.round(Q[0],3))
# per-episode cube cluster extent
ext=np.array([[c[:,0].max()-c[:,0].min(),c[:,1].max()-c[:,1].min()] for c in C])
print("cluster extent x mean %.3f max %.3f ; y mean %.3f max %.3f"%(ext[:,0].mean(),ext[:,0].max(),ext[:,1].mean(),ext[:,1].max()))
d=[np.linalg.norm(c[i]-c[j]) for c in C for i in range(5) for j in range(i+1,5)]
print("min pairwise dist %.3f"%min(d))
