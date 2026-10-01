from geo_lib import *
import sys
seed=int(sys.argv[1]); x,y=float(sys.argv[2]),float(sys.argv[3])
P=Prober(seed,'part0')
z=P.probe(x,y,zs=ZS,zhi=0.145); print('probe lowest',round(z,4))
for zz in np.arange(0.144,0.0995,-0.002):
    if not P.goto([x,y,zz]): break
print('at',np.round(P.pose()[:3],4))
for i in range(3):
    a=np.zeros(11,dtype=np.float32); a[10]=1.0; P._step(a)
    print('open step',i,'ga',P.ap._robot(P.obs)[2],'pose',np.round(P.pose()[:3],4),'term',P.term)
# move arm up
for i in range(5):
    a=np.zeros(11,dtype=np.float32); a[10]=1.0; P._step(a)
print('after', np.round(P.pose()[:3],4),'term',P.term)
