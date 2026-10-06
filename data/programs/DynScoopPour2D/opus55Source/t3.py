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
go((None,2.4,-np.pi/2,0.3,0.09))
ax=X-0.118
go((ax,2.4,None,None,None)); go((ax,0.52+0.3+0.01,None,None,None))
for k in range(4):
    s,*_=env.step(np.array([0.012,0,0,0,-0.0025],dtype=np.float32)); print(rob(s)['x']-X,hook(s))
for k in range(3):
    s,*_=env.step(np.array([0,0.03,0,0,0],dtype=np.float32)); print(hook(s))
print('down through floor')
for k in range(5):
    s,*_=env.step(np.array([0,-0.03,0,0,0],dtype=np.float32)); print(hook(s), rob(s)['y'])
print('left through wall')
for k in range(60):
    s,*_=env.step(np.array([-0.03,0,0,0,0],dtype=np.float32))
    if k%5==0: print(hook(s), rob(s)['x'])
