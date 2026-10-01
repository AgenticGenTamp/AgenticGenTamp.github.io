from geo_lib import *
import sys
seed=int(sys.argv[1])
P=Prober(seed,'part1')
for xy in [tuple(map(float,s.split(','))) for s in sys.argv[2:]]:
    z=P.probe(*xy); print(xy,'lowest z',round(z,4),'pose',np.round(P.pose()[:3],4),flush=True)
