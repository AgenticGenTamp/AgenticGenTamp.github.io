from probe_lib import *
for sd in [0,3,11]:
  for d in [(-1,0),(1,0),(0,-1),(0,1)]:
    o,_=env.reset(seed=sd)
    o=approach(o,*d); print(sd,d,rob(o))
