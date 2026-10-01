import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *
t=time.time()
p=P(1); S=p.S
print(p.cols)
print(p.pick('cuboid_1'), len(S.rew))
R=Rdown(0.0)
yc=0.0
e=p.moveto([1.5,yc,0.3],R,bt=[1.15,yc,0],steps=150); print('mv',e,np.round(grasp_point_world(S),3),p.rel()[0].round(3),len(S.rew))
for z in [0.3,0.15,0.45]:
    print(z,p.push(yc,z,1.55,2.15,R),len(S.rew),flush=True)
print('time',time.time()-t)
