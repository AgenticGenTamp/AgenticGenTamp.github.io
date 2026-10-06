from common import *
from ctrl import *
import sys
env=make_env(); seed=int(sys.argv[1]); s,_=env.reset(seed=seed,options={'object_count':4})
H=hook(s); X,Y=H['x'],H['y']; print(H)
def go(t,maxn=200):
    global s
    for i in range(maxn):
        r=rob(s)
        if reached(r,t): return i
        s,*_=env.step(toward(r,t))
    return -1
press=float(sys.argv[2])
go((None,2.3,0.0,0.3,0.08))
gb=3.36; L=0.3
go((gb-L,2.3,None,None,None)); 
print(go((gb-L,0.5+0.06+0.003,None,None,None)))
print(rob(s))
s,*_=env.step(np.array([0,-press,0,0,0],dtype=np.float32)); print(hook(s))
for k in range(10):
    s,*_=env.step(np.array([-0.01,-0.0,0,0,0],dtype=np.float32)); print(hook(s), rob(s)['x'],rob(s)['y'])
