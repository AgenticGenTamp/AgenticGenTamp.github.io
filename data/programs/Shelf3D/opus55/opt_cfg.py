import numpy as np, itertools
from approach import *
cz=0.05; zc=cz+SHELF2+0.025
res=[]
for pp in [90,75,60,50,45]:
  for pdist in [0.45,0.55,0.65]:
    for rollp in [0]:
      qh,eh=solve(np.array([pdist,0,0.02]),Rpitch(np.radians(pp),0)@Rz(rollp),Q_SEEDS)
      if eh>3e-3: continue
      for pl in [0,15,30,45]:
        for xd in [0.65,0.75,0.85]:
          for rolll in [0,np.pi/2]:
            qp,ep=solve(np.array([xd,0,zc]),Rpitch(np.radians(pl),0)@Rz(rolll),[qh]+Q_SEEDS)
            if ep>3e-3: continue
            d=np.abs(wrap(qp-qh)).max()
            res.append((d,pp,pdist,round(rollp,2),pl,xd,round(rolll,2)))
res.sort()
for r in res[:15]: print(np.round(r,3))
