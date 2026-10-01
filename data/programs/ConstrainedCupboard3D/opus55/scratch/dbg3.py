import sys; sys.path.insert(0,'/sandbox/scratch')
import fprobe
from fprobe import *
p=P(1); S=p.S; S.grip=1.0
for z in [0.15,0.3,0.15]:
    r=bpush(p,0.0,z,1.7,2.3)
    print(z,r)
# now manual: go to z=0.15 pose, push and print trajectory
b0=np.array([1.7-0.62-MX,0,0]); qt=ikh(S,[1.7,0,0.15],b=b0); goto_slow(S,qt=qt,bt=b0,steps=300)
print('start gp',grasp_point_world(S).round(3),'q',S.q().round(2))
for i in range(300):
    a=np.zeros(11); a[10]=1; a[3:10]=np.clip(qt-S.q(),-0.1,0.1); a[0]=0.002; S.step(a)
    if i%25==0: print(i,'gp',grasp_point_world(S).round(3),'base',S.base().round(3),'qerr',np.abs(qt-S.q()).max().round(3))
