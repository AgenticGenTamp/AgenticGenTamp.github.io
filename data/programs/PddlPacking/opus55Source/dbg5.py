import sys, time, numpy as np
from approach import GeneratedApproach, obs_to_flat, Planner, block_info
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv as E
k=int(sys.argv[1]); flat=np.load(sys.argv[2])[int(sys.argv[3])]
env=E([k],[k]); o,info=env.reset(seed=0, options={'object_count':k})
ap=GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(o,info)
P=ap.planner
print('base', np.round(flat[:10],2), 'held', flat[11])
for i,b in enumerate(block_info(flat,k)): print(i, np.round(b['pos'],3), ap._block_done(b))
stats={'ok':0,'fail':0,'checkfail':0}
orig=P.simulate
fails={}
def sim(flat_, acts):
    P.last_fail=None
    r=orig(flat_, acts)
    if not r[0]:
        lf=P.last_fail; key=(lf[1],lf[2]) if lf else 'exact'
        fails[key]=fails.get(key,0)+1
    return r
P.simulate=sim
ap.deadline=time.time()+5
t=time.time()
ap.la_margin=0; ap._alts=[]; ap._alts_keys=[]
r=ap.plan_grasp(flat, [i for i,b in enumerate(block_info(flat,k)) if not ap._block_done(b)], True)
print('res', None if r is None else len(r[0]), time.time()-t, fails)
print('=====')
P.simulate=orig
ob=P.best_path
def bp(flat_, qc, qt, gl, check, pre=None, best_cost=1e9, max_tries=40):
    seqs=P.path_seqs(qc, qt, pre)
    out=[]
    for c,_,seq in seqs[:6]:
        acts=P.seq_actions(qc, seq, gl, 0)
        P.last_fail=None
        ok,fin,term=P.simulate(flat_, acts)
        out.append((c, len(seq)-1, ok, None if ok else (P.last_fail[0], P.last_fail[2], P.last_fail[3].tolist()) if P.last_fail else 'exact', bool(ok and check(fin))))
    print('cand', np.round(qt[:3],2), 'nseq', len(seqs), 'tablecol', P.table_collision(qt), P.table_collision(pre) if pre is not None else None)
    for o_ in out: print('    ', o_)
    return None
P.best_path=bp
ap.deadline=time.time()+5
r=ap.plan_grasp(flat, [7], True)
