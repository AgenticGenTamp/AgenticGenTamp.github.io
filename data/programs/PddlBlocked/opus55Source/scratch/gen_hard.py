import sys, time, numpy as np
sys.path.insert(0,'.')
from approach import *
from approach import _load_lib, _wrap
from robocode.environments.pr2_tamp_blocked_variable_count_env import PR2BlockedVariableCountEnv
env = PR2BlockedVariableCountEnv(design_counts=[0], eval_counts=[0])
seeds=[int(x) for x in sys.argv[2:]]
out=[]
from approach import _rot_err
def box_ik(sim, pt, Rt, q, q0, iters=200, lim=0.98):
    q=q.copy()
    for it in range(iters):
        p,R=sim.tool(q)
        ep=pt-p; er=_rot_err(Rt,R)
        D=conf_delta(q0,q,sim.circ)
        if np.linalg.norm(ep)<1e-4 and np.linalg.norm(er)<2e-3 and np.max(np.abs(D))<lim+0.01:
            return q, True
        J=sim.jacobian(); J[3:]*=0.3
        err=np.concatenate([ep,0.3*er])
        A=J@J.T+0.05**2*np.eye(6)
        dq=J.T@np.linalg.solve(A,err)
        N=np.eye(10)-np.linalg.pinv(J)@J
        dn=(q0+np.clip(D,-lim,lim))-q
        dn[sim.circ]=_wrap(dn[sim.circ])
        dq+=N@dn*0.5
        m=np.max(np.abs(dq))
        if m>0.2: dq*=0.2/m
        q=q+dq; q[sim.circ]=_wrap(q[sim.circ]); q=np.clip(q,sim.lo,sim.hi)
    return q, False

rng=np.random.default_rng(int(sys.argv[1]))
for seed in seeds:
    st,_=env.reset(seed=seed)
    ap=GeneratedApproach(None,None,{}); sim=ap._get_sim(0); flat=sim.sync(st,set_walls=True)
    e=sim.env
    with e.client():
        pl=Planner(sim,time.time()+600); q0=flat[:10].astype(float)
        c=pl.grasp_candidates(q0, e.penned)
        c=[x for x in c if x[0]<=6]
        found=0; t=time.time(); why=[0,0,0]
        for _ in range(int(__import__("os").environ.get("TRIALS","250"))):
            nn,q = c[rng.integers(len(c))]
            e.set_state(flat)
            sim.set_conf(q); pt,Rt = sim.tool()
            # target box
            D=conf_delta(q0,q,sim.circ)
            qref = q0 + np.clip(D, -0.99, 0.99)
            qi = q.copy(); qi[:3]=qref[:3]
            qi[3:] += rng.normal(0,0.1,7)
            qn, ok = box_ik(sim, pt, Rt, qi, q0)
            if not ok: why[0]+=1; continue
            if n_steps(q0,qn,sim.circ)>5: why[1]+=1; continue
            e.set_state(flat)
            if not pl.valid_grasp_conf(qn, e.penned): why[2]+=1; continue
            found+=1; out.append(qn[3:].copy())
        print("seed", seed, "found", found, why, len(c), time.time()-t, flush=True)
np.save(f'scratch/libhard_{sys.argv[1]}.npy', np.array(out).reshape(-1,7))
