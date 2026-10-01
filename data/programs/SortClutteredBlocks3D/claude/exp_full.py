import numpy as np, sys
from env_client import make_env
import approach as A
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
bins={n:A.obj_pos(obs,n) for n in obs.get_object_names() if n.startswith('bin_')}
def placed(obs,name):
    c=A.obj_pos(obs,name); bp=ap._bin_for(name)
    return abs(c[0]-bp[0])<0.045 and abs(c[1]-bp[1])<0.045 and c[2]>0.405
rs={}; T=0; term=None
for rnd in range(6):
    t=0
    while ap.idx < len(ap.order) and t<1200 and T<6000:
        a=ap.get_action(obs); obs,r,te,tr,i=env.step(a); t+=1; T+=1
        r=float(r); rs[round(r,6)]=rs.get(round(r,6),0)+1
        if abs(r+1)>1e-9: print('REWARD!',seed,T,r,flush=True)
        if te or tr: term=(T,te,tr); break
    if term: break
    bad=[n for n in ap.order if not placed(obs,n)]
    st={n:np.round(A.obj_pos(obs,n),3).tolist() for n in sorted(ap._cubes(obs))}
    print(seed,'round',rnd,'T',T,'bad',bad,st,flush=True)
    if not bad: break
    ap.order=bad; ap.idx=0; ap.phase='app'; ap.t=0; ap.bias=np.zeros(3); ap.target=None
# after all placed, step some more to see reward
for k in range(30):
    obs,r,te,tr,i=env.step(np.zeros(11,dtype=np.float32)); r=float(r)
    rs[round(r,6)]=rs.get(round(r,6),0)+1
    if te or tr: term=(T+k,te,tr); break
print('SEED',seed,'rewards',rs,'term',term,flush=True)
env.close()
