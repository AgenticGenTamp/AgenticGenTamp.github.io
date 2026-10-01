from probe_lib import *
import sys
N=int(sys.argv[1]); pr=P(0)
xy=(0.3,-0.12); top=[xy[0],xy[1],0.56]
pr.goto(pr.ik(top))
pr.grip=1.0
for i in range(N): pr.step(np.r_[pr.QT*0,[1.0]][ [0,1,2]+list(range(10,11)) ] if False else np.r_[np.zeros(10),1.0])
p,why,qe=pr.line(top,[xy[0],xy[1],0.36],ds=0.002,qthr=0.006)
print(N,'contact fk z',pr.fk()[2],why)
