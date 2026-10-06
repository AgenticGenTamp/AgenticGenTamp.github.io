import sys, numpy as np
from common import make_env, hook, rob
from ctrl import toward, reached
seed,alpha,gap,cnt,pdx,marg=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])
env=make_env(); s,info=env.reset(seed=seed,options={'object_count':cnt})
H=hook(s); X,Y=H['x'],H['y']
D=-np.pi/2
def run(seq):
    global s
    for g in seq:
        for k in range(200):
            R=rob(s)
            if reached(R,g): break
            s,_,_,_,_=env.step(toward(R,g))
R=rob(s)
px=X-0.5-0.1
run([(R["x"],2.6,D,0.4,0.09),(px,2.6,D,0.4,0.09),(px,0.9,D,0.4,0.09),(px,0.56,D,0.4,0.09)])
print('pre',{k:round(v,3) for k,v in hook(s).items()})
for k in range(40):
    R=rob(s)
    if hook(s)['x']>3.497 or R['x']>3.28: break
    s,_,_,_,_=env.step(toward(R,(R['x']+0.03,0.56,D,0.4,0.09)))
print('pushed',{k:round(v,3) for k,v in hook(s).items()})
R=rob(s); run([(R['x'],0.9,D,0.4,0.09)])
H=hook(s); X,Y=H['x'],H['y']
th=-np.pi/2+alpha; c,si=np.cos(th),np.sin(th)
def rot(v): return np.array([c*v[0]-si*v[1], si*v[0]+c*v[1]])
P=np.array([X-0.05,Y+0.5])
G=P-rot((0.03,gap/2+marg)); B=G-rot((0.4,0))
run([(B[0]-0.03,B[1]+0.15,th,0.4,gap),(B[0],B[1],th,0.4,gap)])
for k in range(12):
    s,_,_,_,_=env.step(np.array([pdx,0,0,0,-0.002],dtype=np.float32))
    if hook(s)['held']>0.5: print('HELD at',k); break
print("final",{k:round(v,3) for k,v in rob(s).items()}, {k:round(v,3) for k,v in hook(s).items()})
