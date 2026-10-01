import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *
seed=int(sys.argv[1]); k=None if sys.argv[2]=='n' else int(sys.argv[2]); BX=float(sys.argv[3])
pts=[tuple(float(v) for v in s.split(',')) for s in sys.argv[4].split(';')]
p=P(seed,k); S=p.S; S.grip=1.0
yaw=0.0
for (x,y,ztop) in pts:
    goto_slow(S, bt=np.array([BX,y,0]), steps=150, vmax=0.08); bb=S.base()
    if abs(bb[0]-BX)>0.01: print('BASE BLOCKED',bb)
    r=probe_dir(p,[x,y,ztop],[0,0,-1],ztop-0.005,Rdown(np.pi/2),dx=0.008,thr=0.007,vmax=0.06)
    print('P x %.3f y %.3f contact %s steps %d'%(x,y,r,len(S.rew)),flush=True)
print('LOG',LOG)
