import numpy as np
from kin import ik
from planner import make_plan
def remove_blocker(S, drop=(4.62,-0.45)):
    blk=S.block('blocker'); g0=S.block('green0')
    wps=make_plan(S.base(),S.q(),blk,g0,g0[2])
    for w in wps:
        for bb,qq in w['configs']: S.moveto(bb,qq,maxd=0.1)
        if w['grip']!=0: S.step(np.r_[np.zeros(10),w['grip']])
        if w['grip']<0: break
    print('grasp',S.rget('grasp_active'),[w['name'] for w in wps])
    d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d)
    b=S.base(); ok=True
    for z in [0.9,1.0,1.1]:
        _,q,e=ik(np.r_[blk[:2],z],d,b,S.q(),free_base=False); ok&=S.moveto(b,q,maxd=0.1)
    _,q,e=ik(np.r_[drop,1.0],np.array([1.,0,0]),b,S.q(),free_base=True); ok&=S.moveto(b if False else _,q,maxd=0.1)
    for _ in range(3): S.step(np.r_[np.zeros(10),1.0])
    for _ in range(5): S.step(np.zeros(11))
    return S.block('blocker'), S.rget('grasp_active') if hasattr(S,'rget') else None
if __name__=='__main__':
    from env_client import make_env; from envutil import Sim
    import sys
    env=make_env(); S=Sim(env,env.reset(seed=int(sys.argv[1]))[0])
    print(remove_blocker(S), S.steps, S.base().round(2))
