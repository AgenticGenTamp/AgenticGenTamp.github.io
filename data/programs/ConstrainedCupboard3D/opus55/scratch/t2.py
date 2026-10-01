import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *
p=P(1); S=p.S
p.pick('cuboid_1'); R=Rdown(0.0)
for yc in [0.6, 0.1, 0.05, 0.3]:
    p.moveto([1.5,yc,0.3],R,bt=[1.15,yc,0],steps=150)
    for z in [0.3]:
        print(yc,z,p.push(yc,z,1.55,2.3,R),len(S.rew),flush=True)
