import sys, numpy as np, mujoco
from simutil import make_env
from approach import GeneratedApproach
N=int(sys.argv[1]); sd=int(sys.argv[2]); steps=[int(x) for x in sys.argv[3].split(',')]
env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
def find_sim(e):
    e=env._current_backend if not hasattr(env,"_object_centric_env") else e
    for _ in range(10):
        if hasattr(e,'_object_centric_env'): e=e._object_centric_env
        if hasattr(e,'_robot_env'): return e._robot_env.sim
        e=getattr(e,'env',None) or getattr(e,'_env',None) or getattr(e,'_constant_env',None)
        if e is None: break
    return None
sim=None
for k in range(max(steps)+1):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    if k in steps:
        if sim is None:
            sim=find_sim(env)
            if sim is None:
                print(type(env._current_backend), [x for x in vars(env._current_backend)][:20]); break
        m=sim.model.mj_model; d=sim.data.mj_data
        names=set()
        for i in range(d.ncon):
            c=d.contact[i]
            b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]); b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2])
            if 'cube' in str(b1) and 'cube' in str(b2): continue
            names.add((b1,b2))
        print(k,sorted(names)[:30])
        print('qfrc_act arm', np.round(d.qfrc_actuator[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot_joint_1')]:][:7],1))
        for nm in ['robot_forearm_link','robot_spherical_wrist_1_link','robot_spherical_wrist_2_link','robot_bracelet_link','bin_green_0','bin_yellow_0']:
            b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,nm); print(nm, np.round(d.xpos[b],3))
        for i in range(d.ncon):
            c=d.contact[i]; n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]); n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2])
            if 'wrist' in n1+n2 or 'forearm' in n1+n2: print('  contact',n1,n2,np.round(c.pos,3))

    if term: break