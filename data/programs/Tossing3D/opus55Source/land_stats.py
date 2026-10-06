import numpy as np, sys, json
from common import make_env, robot, obj
from approach import GeneratedApproach
env=make_env([1,2])
counts=json.loads(sys.argv[1]); seeds=json.loads(sys.argv[2])
appr=GeneratedApproach(env.action_space, env.observation_space, {})
for n in counts:
  for seed in seeds:
    s,info=env.reset(seed=seed, options={'object_count':n}); appr.reset(s,info)
    prev={c:obj(s,c) for c in appr._cubes()}
    for i in range(400+400*n):
        a=appr.get_action(s); s,r,term,_,_=env.step(a)
        b=obj(s,'bin_0'); base,_=robot(s)
        for c in appr._cubes():
            cur=obj(s,c); p=prev[c]
            if p[2]>0.2 and cur[2]<=0.2 and p[9]<0 and p[0]>1.5:
                aa=-4.905; bb=p[9]; cc=p[2]-0.2; t=(-bb-np.sqrt(bb*bb-4*aa*cc))/(2*aa)
                x=p[:2]+p[7:9]*t
                d=np.array(p[7:9])/np.linalg.norm(p[7:9])
                rel=x-b[:2]; print(f'n={n} seed={seed} {c} cross0.2 along={rel@d:.3f} lat={rel@np.array([-d[1],d[0]]):.3f} binmove? {b[:2].round(3)}')
            prev[c]=cur
        if term: break
    print(f'n={n} seed={seed} term={term} final', [ (obj(s,c)[:2]-obj(s,"bin_0")[:2]).round(3).tolist() for c in appr._cubes()])
