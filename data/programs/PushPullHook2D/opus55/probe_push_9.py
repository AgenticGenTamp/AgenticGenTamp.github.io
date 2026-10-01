from lib import *
env=make_env()
B=[];T=[]
for s in range(40):
    o,_=env.reset(seed=s); B.append(o[20:22]); T.append(o[29:31])
B=np.array(B);T=np.array(T)
print('btn min',B.min(0),'max',B.max(0)); print('tgt min',T.min(0),'max',T.max(0)); print('dist min/max',np.linalg.norm(B-T,axis=1).min(),np.linalg.norm(B-T,axis=1).max())
