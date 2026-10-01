import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *
p=P(1); S=p.S; R=Rdown(np.pi/2)
p.pick('cuboid_1'); g=grasp_point_world(S)
print(moveto_slow(p,[g[0],g[1],0.4],R), p.rel()[1].round(2))
print(moveto_slow(p,[1.6,0,0.35],R,bt=[1.3,0,0]), grasp_point_world(S).round(3), S.base().round(3))
for x in [1.6,1.65,1.7,1.75]:
    e,n=moveto_slow(p,[x,0,0.4],R); print(x,e,n,grasp_point_world(S).round(3),p.rel()[1].round(3))
from kin import fk_all
fr,T=fk_all(S.q(),TOOL); print('q',S.q().round(2)); print('joint z (arm frame)',[round(f[2,3]+MZ,2) for f in fr],'x',[round(f[0,3]+MX+S.base()[0],2) for f in fr])
