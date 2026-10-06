import sys, numpy as np
from common import make_env, hook, rob
from ctrl import toward, reached
seed,alpha,gap,cnt=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
env=make_env(); s,info=env.reset(seed=seed,options={'object_count':cnt})
H=hook(s); X,Y=H['x'],H['y']
th=-np.pi/2+alpha; c,si=np.cos(th),np.sin(th)
def rot(v): return np.array([c*v[0]-si*v[1], si*v[0]+c*v[1]])
P=np.array([X-0.05,Y+0.5])
G=P-rot((0.03,gap/2+0.024)); B=G-rot((0.4,0))
print('X',round(X,3),'base',B, 'hook local y', rot((0,0)) , np.dot(np.array([X,Y])-G, [-si,c]))
R=rob(s)
seq=[(R['x'],max(R['y'],2.3),th,0.4,gap),(B[0]-0.03,B[1]+0.15,th,0.4,gap),(B[0],B[1],th,0.4,gap)]
for g in seq:
    for k in range(200):
        R=rob(s)
        if reached(R,g): break
        s,_,_,_,_=env.step(toward(R,g))
print({k:round(v,3) for k,v in rob(s).items()}, {k:round(v,3) for k,v in hook(s).items()})
for k in range(12):
    s,_,_,_,_=env.step(np.array([0.004,0,0,0,-0.002],dtype=np.float32))
    if hook(s)['held']>0.5: print('HELD at',k); break
print({k:round(v,3) for k,v in rob(s).items()}, {k:round(v,3) for k,v in hook(s).items()})
