import numpy as np, sys
from env_client import make_env
import approach as A
seed=int(sys.argv[1]); oc=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
rs=set(); T=0; term=None
def dump(tag):
    d={n:np.round(A.obj_pos(obs,n),3).tolist() for n in sorted(obs.get_object_names()) if n.startswith(('cube','bin'))}
    print(tag,T,d,flush=True)
for rnd in range(8):
    t=0
    while ap.idx < len(ap.order) and t<1200 and T<5000:
        a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); t+=1; T+=1; rs.add(round(float(r),6))
        if te or tr:
            term=(T,te,tr); dump('TERMSTATE'); print('terminal reward',r,flush=True); break
    if term: break
    ap.order=[n for n in sorted(ap._cubes(obs),key=lambda x:int(x[4:]))]; ap.idx=0; ap.phase='app'; ap.t=0; ap.bias=np.zeros(3); ap.target=None
    dump('round%d'%rnd)
print('SEED',seed,oc,'distinct rewards',sorted(rs),'term',term,flush=True)
env.close()
