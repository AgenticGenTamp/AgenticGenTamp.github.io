import numpy as np, sys
from env_client import make_env
import approach as A
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
rs=set(); prev=None
for T in range(1,1501):
    obs,r,te,tr,i=env.step(ap.get_action(obs)); rs.add(round(float(r),6))
    cub=sorted(n for n in obs.get_object_names() if n.startswith('cube'))
    ok=[n for n in cub if ap._placed(n, A.obj_pos(obs,n))]
    if len(ok)!=prev: print(seed,T,'sorted',len(ok),ok,'r',r,flush=True); prev=len(ok)
    if te or tr:
        print(seed,'TERM',T,'te',te,'tr',tr,'r',r,'sorted',ok,flush=True)
        print({n:np.round(A.obj_pos(obs,n),3).tolist() for n in sorted(obs.get_object_names()) if n.startswith(('cube','bin'))},flush=True)
        break
print(seed,'rewards',sorted(rs),flush=True)
