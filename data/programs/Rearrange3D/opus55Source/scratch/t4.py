import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
from scipy.spatial.transform import Rotation as R
oc=env._kinder_env._object_centric_env; bowl=oc._objects_dict['bowl_0']
def inreg(p): return bowl.check_in_region(np.array(p,dtype=np.float32),'bowl_0_region',re)
seed=int(sys.argv[1])
def body(m,name): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
def trial(dx,dy,tilt_deg,z0):
    h=H(seed); h.step(60); m,d=h.m,h.d
    dp=h.drink(); b=body(m,"bowl_0"); qa=m.jnt_qposadr[m.body_jntadr[b]]; da=m.jnt_dofadr[m.body_jntadr[b]]
    # tilt axis perpendicular to offset: bowl low side away from drink
    u=np.array([dx,dy,0.]); u/=np.linalg.norm(u); ax=np.cross([0,0,1.],u)  # rotating about ax by +ang lifts -u side? check
    rot=R.from_rotvec(ax*np.radians(tilt_deg)); qq=rot.as_quat()
    d.qpos[qa:qa+3]=[dp[0]+dx,dp[1]+dy,z0]; d.qpos[qa+3:qa+7]=[qq[3],*qq[:3]]; d.qvel[da:da+6]=0
    mujoco.mj_forward(m,d)
    while True:
        pen=[d.contact[i].dist for i in range(d.ncon) if b in (m.geom_bodyid[d.contact[i].geom1],m.geom_bodyid[d.contact[i].geom2]) and d.contact[i].dist<-0.0005]
        if not pen: break
        d.qpos[qa+2]+=0.001; mujoco.mj_forward(m,d)
    z0=d.qpos[qa+2]
    for t in range(NST):
        h.step()
        if TR and t%15==0: print('   t',t,np.round(d.xpos[b],3), round(float(np.degrees(np.arccos(d.xmat[b][8]))),1))
    bp=d.xpos[b].copy(); zax=d.xmat[b].reshape(3,3)[:,2]; th=np.degrees(np.arccos(zax[2]))
    dr=d.xpos[body(m,"boxed_drink_0")]
    print(f"dx{dx:+.3f} dy{dy:+.3f} tilt0 {tilt_deg} z0 {z0:.3f} -> bowl {np.round(bp,3)} tilt {th:.1f} drink {np.round(dr,3)} IN {inreg(dr)} bbox {np.round(bowl.region_objects['bowl_0_region'][0].bbox,3)}")
    return h
TR=False; NST=300
for dx,dy,tilt,z0 in eval(sys.argv[2] if len(sys.argv)>2 else "[(0.07,0,15,0.50),(0.08,0,15,0.50),(0.09,0,14,0.497),(0.10,0,13,0.495)]"):
    h=trial(dx,dy,tilt,z0)
if len(sys.argv)>3:
    m,d=h.m,h.d; b=body(m,"bowl_0")
    for i in range(d.ncon):
        c=d.contact[i]; b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
        if b in (b1,b2):
            o=b2 if b1==b else b1; print("bowl contact", mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,o), mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1 if b1!=b else c.geom2), np.round(c.pos,3), np.round(c.frame[:3],2))
    print("zaxis", np.round(d.xmat[b].reshape(3,3)[:,2],3)); print("STATE", list(np.round(env.get_state() if hasattr(env,'get_state') else h.obs,4))[:50])
