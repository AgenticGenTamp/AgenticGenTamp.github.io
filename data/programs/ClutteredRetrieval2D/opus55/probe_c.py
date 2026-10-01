from probe_lib import *
for sd in range(15):
    o,_=env.reset(seed=sd)
    obs_=[n for n in names(o) if n.startswith('obs')]
    xs=[g(o,n,'x') for n in obs_]; ys=[g(o,n,'y') for n in obs_]
    print(sd,'R',rob(o)[:3],'B',obj(o,'target_block'),'G',obj(o,'target_region'),'obs x[%.2f,%.2f] y[%.2f,%.2f]'%(min(xs),max(xs),min(ys),max(ys)))
