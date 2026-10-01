import sys, numpy as np, stages, planner
from env_client import make_env
from envutil import Sim
from stages import geom, base_candidates, green_chain_from, GRASP_DZ
s=int(sys.argv[1]); env=make_env(); S=Sim(env,env.reset(seed=s)[0])
blk=S.block('blocker'); g0=S.block('green0'); d,perp,z=geom(blk,g0,g0[2])
print('d',d.round(2),'g0',g0[:2].round(3))
cg=np.array([g0[0],g0[1],z]); tgt=cg-0.08*d
for b,q in base_candidates(np.array([4.75,1.02,3.17]),tgt,d,planner.TABLE):
    print('cand',b.round(2),q.round(2))
    import stages as st
    # replicate chain with debug
    UP=st.UP; lift=z+0.16
    seq=[("g_pre", cg-0.12*d+0.06*UP,0.0,d),("axis",cg-0.12*d,0.0,d),("grasp_g",cg-st.GRASP_BACK*d,-1.0,d),("out",cg+(lift-z)*UP,0.0,d)]
    for k in range(1,len(seq)+1):
        r=st.solve_chain(b,q,None,None,seq[:k])
        if r is None: print('   fail at',seq[k-1][0]); break
    else:
        print('   chain ok; full:', green_chain_from(b,q,d,cg,z,planner.TABLE,planner.PLATE,[blk],True) is not None)
