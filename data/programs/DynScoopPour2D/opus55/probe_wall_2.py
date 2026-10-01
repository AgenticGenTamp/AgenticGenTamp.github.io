from probe_wall_1 import *
import sys, itertools
seed=int(sys.argv[1]); arms=[float(v) for v in sys.argv[2].split(',')]
for arm in arms:
  for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    for gap in [0.15,0.2,0.25]:
        p=P(seed); rd,r,h=attempt(p,a,arm,gap); p.env.close()
        print(seed,'a',a,'arm',arm,'gap',gap,'desc',rd['x'],rd['y'],'held',h['held'],'hook',h['x'],h['theta'],flush=True)
