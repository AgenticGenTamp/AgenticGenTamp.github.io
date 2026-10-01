from geo_lib import *
import sys
P=Prober(int(sys.argv[1]), sys.argv[2] if len(sys.argv)>2 else None)
print('part',P.pname,'pose',np.round(P.pose(),4),'steps',P.nsteps)
for xy in [(0.3,0.0),(0.3,0.12),(0.3,-0.12),(0.2,0),(0.4,0)]:
    print(xy, round(P.probe(*xy),4), P.nsteps)
