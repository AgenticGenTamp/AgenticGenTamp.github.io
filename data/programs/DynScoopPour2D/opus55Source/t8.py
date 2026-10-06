import sys, numpy as np
from common import make_env, hook, rob
from ctrl import toward, reached
seed,alpha,press=int(sys.argv[1]),float(sys.argv[2]),sys.argv[3]
env=make_env(); s,info=env.reset(seed=seed,options={'object_count':1})
H=hook(s); X,Y=H['x'],H['y']
th=-np.pi/2+alpha
c,si=np.cos(th),np.sin(th)
def base_for(back):
    rx,ry=0.93+back,0.123
    return (X-(c*rx-si*ry), Y-(si*rx+c*ry))
seq=[]
R=rob(s)
seq.append((R['x'],max(R['y'],2.3),th,0.4,0.09))
b=base_for(0.15); seq.append((b[0],b[1],th,0.4,0.09))
b=base_for(0.0); seq.append((b[0],b[1],th,0.4,0.09))
print('X',X,'base',b)
t=0
for g in seq:
    for k in range(200):
        R=rob(s)
        if reached(R,g): break
        s,_,_,_,_=env.step(toward(R,g)); t+=1
    print('reached',g, {k:round(v,3) for k,v in rob(s).items()}, {k:round(v,3) for k,v in hook(s).items()})
pr=[float(v) for v in press.split(',')]
for k in range(12):
    s,_,_,_,_=env.step(np.array(pr,dtype=np.float32))
    H=hook(s)
    if H['held']>0.5: print('HELD at',k); break
print({k:round(v,3) for k,v in rob(s).items()}, {k:round(v,3) for k,v in hook(s).items()})
