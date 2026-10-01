import sys; sys.path.insert(0,'/sandbox/scratch')
from fprobe import *
p=P(1); S=p.S; S.grip=1.0
goto_slow(S, bt=np.array([0.9, 0, 0]), steps=100, vmax=0.1)
print('q0',S.q().round(2),S.base().round(3))
for tgt,b in [([1.55,0,0.35],None),([1.62,0,0.35],[1.35,0,0]),([1.72,0,0.4],None),([1.72,0,0.2],None)]:
    qt=ikh(S,tgt,b=b); print('qt',None if qt is None else qt.round(2))
    if qt is None: continue
    n=goto_slow(S,qt=qt,bt=None if b is None else np.array(b),steps=300,vmax=0.06)
    print(tgt,'n',n,'q',S.q().round(2),'err',np.abs(S.q()-qt).max().round(3),'gp',grasp_point_world(S).round(3),'base',S.base().round(3))
