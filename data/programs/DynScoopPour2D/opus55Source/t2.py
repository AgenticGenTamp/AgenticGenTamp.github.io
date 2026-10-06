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
print(go((None,2.4,-np.pi/2,0.3,0.09)))
ax=X-0.118
print(go((ax,2.4,None,None,None)))
print(go((ax,0.52+0.3+0.01,None,None,None)))
print(rob(s),hook(s))
s,*_=env.step(np.array([0.01,0,0,0,-0.01])); print(rob(s),hook(s))
s,*_=env.step(np.array([0,0.03,0,0,0])); print(rob(s),hook(s))
