import sys, os, json, collections, numpy as np
import exp_approach as A
from env_client import make_env
seed=int(sys.argv[1]); tag=sys.argv[2]
cfg=json.loads(sys.argv[3]) if len(sys.argv)>3 else {}
for k,v in cfg.items(): setattr(A,k,v)
if "G_OPEN" in cfg and "OPEN_GAP" not in cfg: A.OPEN_GAP=0.085*(1-A.G_OPEN)
env=make_env(); obs,info=env.reset(seed=seed,options={"object_count":20})
ap=A.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
trans=collections.Counter(); last=None; recs=[]; cur=None
for t in range(1000):
    a=ap.get_action(obs); ph=ap.phase
    if last!=ph:
        trans[f"{last}>{ph}"]+=1
        if ph=="goto_pick":
            n=ap.target; k=ap.sel_k
            clr=[float(ap._axis_clearance(n,ap.phi)), float(ap._axis_clearance(n,ap.phi+np.pi/2))]
            cur=dict(n=n,k=k,clr=clr[0],clr_other=clr[1],p0={m:o["p"].copy() for m,o in ap.cubes.items() if m!=n})
        if last=="close" and cur is not None:
            # neighbour disturbance during descend/close
            mv=max((np.linalg.norm(ap.cubes[m]["p"]-p) for m,p in cur["p0"].items()),default=0)
            cur["nbr_mv"]=float(mv)
        if last in("lift","descend") and ph in("select","putback","transport") and cur is not None:
            cur["out"]=f"{last}>{ph}"; cur.pop("p0",None); recs.append(cur); cur=None
    last=ph
    obs,r,term,trunc,info=env.step(a)
    if term: break
ap._parse(obs)
placed=sum(ap._in_bin(n) for n in ap.cube_names)
res=dict(seed=seed,tag=tag,placed=placed,steps=t+1,trans={k:v for k,v in trans.items() if k in("lift>select","lift>putback","descend>select","lift>transport")},recs=recs)
open(f"exp_res_{tag}_{seed}.json","w").write(json.dumps(res))
print(tag,seed,placed,res["trans"])
