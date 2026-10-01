from geo_lib import *
import sys
seed=int(sys.argv[1])
P=Prober(seed,'part1')
for s in sys.argv[2:]:
    x,y=map(float,s.split(','))
    P.goto([x,y,0.145]); print((x,y),'at hi',np.round(P.pose()[:3],4))
    for z in ZS:
        ok=P.goto([x,y,z])
        if not ok: print('  rej at',round(z,4),'pose',np.round(P.pose()[:3],4)); break
    else: print('  reached',np.round(P.pose()[:3],4))
    P.goto([x,y,0.145])
