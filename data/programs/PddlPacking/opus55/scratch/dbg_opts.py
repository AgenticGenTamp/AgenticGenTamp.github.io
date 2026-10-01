import sys; sys.path.insert(0,'.')
import numpy as np, time, math
from env_client import make_env
from approach import *
from kin import ik_down, Q0
seed=int(sys.argv[1]); N=int(sys.argv[2]); bn=sys.argv[3]
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(N):
    a=ap.get_action(obs); obs,*_=env.step(a)
cfg,blocks,holding,held,gtf,gq=ap._parse(obs)
w=World(blocks); b=blocks[bn]
print(cfg.round(2), b)
bases=block_bases(b[0],b[1],dense=True)
print("nbases",len(bases))
nik=0; nok=0; ncol=0
for base in bases:
    c=ap._ik_cfg(base,np.array([b[0],b[1],b[2]+GRASP_DZ]),Q0,yaw=b[3])
    nik+=1
    if c is None: continue
    nok+=1
    oks=[ap._config_ok(np.r_[c[:9],wrap(c[9]+k*math.pi/2)],w,exclude=bn,margin=0.0) for k in range(4)]
    if not any(oks): ncol+=1; print("col", np.round(base,2))
    else: print("ok", np.round(base,2))
print(nik,nok,ncol)
t=time.time(); o=ap._grasp_options(cfg,bn,b,w,extra=True); print("opts",len(o), time.time()-t)
o=ap._grasp_options(cfg,bn,b,w)
print("normal opts", len(o))
for sc,gc in o[:6]:
    print(sc, gc[:3].round(2), "direct", ap._path_ok(cfg,gc,w), "route", ap._route(cfg,gc,w,None,False) is not None)
import approach
gc=o[0][1]
# monkeypatch seq_ok to print
orig=ap._seq_ok
def so(seq,world,held,closed):
    print("seq bases",[np.round(c[:3],2).tolist() for c in seq])
    for i in range(len(seq)-1):
        if i>0: print(" node ok", ap._config_ok(seq[i],world,None,held,closed), "base_ok", base_ok(seq[i][:3]))
        n=nsteps(seq[i],seq[i+1]); d=cfg_diff(seq[i],seq[i+1])
        for k in range(1,n):
            c=seq[i]+d*k/n
            if not ap._config_ok(c,world,None,held,closed): print("  bad interm",i,k,np.round(c[:3],2), base_ok(c[:3]))
    return orig(seq,world,held,closed)
ap._seq_ok=so
print(ap._route(cfg,gc,w,None,False))
