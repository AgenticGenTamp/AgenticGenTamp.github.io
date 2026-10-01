from probe_lib import *
import sys
pr=P(0); pr.grip=float(sys.argv[1])
for xy in [(0.3,-0.12),(0.45,0.0),(0.2,-0.2)]:
    top=[xy[0],xy[1],0.62]
    pr.goto(pr.ik(top))
    print('at top fk',pr.fk(),'qerr',np.abs(pr.obs[96:103]-pr.QT).max())
    p,why,qe=pr.line(top,[xy[0],xy[1],float(sys.argv[2])],ds=0.001,qthr=0.006)
    print(xy,'contact cmd z',p[2],'fk',pr.fk(),why,qe)
    pr.goto(pr.ik(top))
