import numpy as np
from kinder.envs.kinematic2d.clutteredretrieval2d import ObjectCentricClutteredRetrieval2DEnv as ClutteredRetrieval2DEnv
from kinder.envs.utils import state_2d_has_collision
from approach import *
rng=np.random.default_rng(0)
mism=0; tot=0; near=0
for seed in range(15):
    e=ClutteredRetrieval2DEnv(num_obstructions=8)
    s,_=e.reset(seed=seed)
    ap=GeneratedApproach(None,None,{})
    for _ in range(300):
        st=s.copy()
        robot=[o for o in st if o.type.name=='crv_robot'][0]
        tb=[o for o in st if o.name=='target_block'][0]
        tc=np.array([st.get(tb,'x'),st.get(tb,'y')])
        q=np.array([tc[0]+rng.normal(0,0.25),tc[1]+rng.normal(0,0.25),rng.uniform(-np.pi,np.pi),rng.uniform(0.1,0.2)])
        for f,v in zip(['x','y','theta','arm_joint'],q): st.set(robot,f,v)
        full=st.copy(); full.data.update(e.initial_constant_state.data)
        others=set(o for o in full if o!=robot)
        env_col=state_2d_has_collision(full,{robot},others,{})
        r,rects=ap._parse(st)
        chk=Checker(ap._robot_model(r),ap._obstacles(rects,set()),[])
        mine=not chk.free(q,margin=0.0)
        mine_m=not chk.free(q,margin=MARGIN)
        tot+=1
        if mine!=env_col: mism+=1; print("mismatch",seed,q,env_col,mine)
        if env_col and not mine_m: print("UNSAFE",seed,q)
print("mismatch",mism,"/",tot)
# compare batch with scalar
import time
mm=0; T1=T2=0
for seed in range(10):
    e=ClutteredRetrieval2DEnv(num_obstructions=15); s,_=e.reset(seed=seed)
    r,rects=ap._parse(s); m=ap._robot_model(r)
    tgt=rects['target_block']
    spec=ap._spec_for(m, ap._grasp_cfgs(tgt,m)[0], tgt)
    chk=Checker(m,ap._obstacles(rects,{'target_block'}),[spec],extra_suction_check=True)
    tc=np.array([tgt['x'],tgt['y']])
    Q=np.stack([tc[0]+rng.normal(0,0.3,500),tc[1]+rng.normal(0,0.3,500),rng.uniform(-np.pi,np.pi,500),rng.uniform(0.1,0.2,500)],1)
    t=time.time(); a=np.array([chk.free_scalar(q) for q in Q]); T1+=time.time()-t
    t=time.time(); b=batch_free(chk,Q); T2+=time.time()-t
    mm+=(a!=b).sum()
print("batch mismatches",mm,"scalar",T1,"batch",T2)
mm=0;T3=0
for seed in range(10):
    e=ClutteredRetrieval2DEnv(num_obstructions=15); s,_=e.reset(seed=seed)
    r,rects=ap._parse(s); m=ap._robot_model(r)
    tgt=rects['target_block']
    spec=ap._spec_for(m, ap._grasp_cfgs(tgt,m)[0], tgt)
    chk=Checker(m,ap._obstacles(rects,{'target_block'}),[spec],extra_suction_check=True)
    tc=np.array([tgt['x'],tgt['y']])
    Q=np.stack([tc[0]+rng.normal(0,0.3,500),tc[1]+rng.normal(0,0.3,500),rng.uniform(-np.pi,np.pi,500),rng.uniform(0.1,0.2,500)],1)
    a=np.array([chk.free_scalar(q) for q in Q])
    t=time.time(); b=np.array([fast_free(chk,q) for q in Q]); T3+=time.time()-t
    mm+=(a!=b).sum()
print("fast mismatches",mm,"fast time",T3)
