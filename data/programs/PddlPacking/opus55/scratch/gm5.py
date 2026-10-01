# usage: gm5.py grip zt ys zs ye ze [yawoff_deg]  -> advance from (ys,zs) toward (ye,ze) (tool y/z offsets from block centre) until rejected
import sys, time; sys.path.insert(0,'scratch')
from gm_lib import *
a=[float(x) for x in sys.argv[1:]]
grip,zt,ys,zs,ye,ze=a[:6]; yoff=a[6] if len(a)>6 else 0.0
r=R(27,(-0.47,0.22,0.0))
if grip<0: r.grip(-1.0)
bc,byaw=block(r.o); yaw=byaw+math.radians(yoff); c,s=math.cos(yaw),math.sin(yaw)
Y=np.array([c,s,0.]); Z=np.array([s,-c,0.])
P=lambda y,z,h=zt: np.array([bc[0],bc[1],h])+y*Y+z*Z
assert r.move_tool(P(ys,zs,0.95),yaw),"hi"
assert r.move_tool(P(ys,zs),yaw),"start"
ps=P(ys,zs); pe=P(ye,ze); L=np.linalg.norm(pe-ps); u=(pe-ps)/L; t=0.0
for st in (0.01,0.002,0.0005,0.0002):
    while t+st<=L+1e-9 and r.try_tool(ps+u*(t+st),yaw): t+=st
f=t/L
print("grip %g zt %.4f yawoff %g start(%.3f,%.3f)->end(%.3f,%.3f): last free at y=%.4f z=%.4f%s"%(grip,zt,yoff,ys,zs,ye,ze,ys+f*(ye-ys),zs+f*(ze-zs)," (reached end)" if t>=L-1e-6 else ""),flush=True)
