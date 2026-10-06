import sys, numpy as np, mujoco
from simutil import make_env, obj
from approach import GeneratedApproach, quat_R
N=int(sys.argv[1]); sd=int(sys.argv[2]); T=int(sys.argv[3])
env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
for k in range(T):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
sim=env._current_backend._object_centric_env._robot_env.sim if hasattr(env._current_backend,'_object_centric_env') else None
m=sim.model.mj_model; d=sim.data.mj_data
Y=obj(s,'bin_yellow_0'); Rb=quat_R(*Y[3:])
for i in range(N):
    c=obj(s,'cube_%d'%i)
    if c[2]>0.6:
        print('cube',i,np.round(c[:3],3),'local',np.round(Rb.T@(c[:3]-Y[:3]),3))
        for j in range(d.ncon):
            cc=d.contact[j]; n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[cc.geom1]); n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[cc.geom2])
            if 'cube_%d'%i in (n1,n2): print('  ',n1,n2,mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,cc.geom1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,cc.geom2))
