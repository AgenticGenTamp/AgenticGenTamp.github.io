import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
from scipy.spatial.transform import Rotation as R
oc=env._kinder_env._object_centric_env; bowlo=oc._objects_dict['bowl_0']
def inreg(p): return bowlo.check_in_region(np.array(p,dtype=np.float32),'bowl_0_region',re)
def body(m,n): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
DBG=len(sys.argv)>4; seed=int(sys.argv[1]); S0=int(sys.argv[3]) if len(sys.argv)>3 else 60
def trial(uang, th_deg, psi_deg, ov, gap=None, NST=300):
    h=H(seed); h.step(S0); m,d=h.m,h.d; dp=h.drink()
    b=body(m,"bowl_0"); qa=m.jnt_qposadr[m.body_jntadr[b]]; da=m.jnt_dofadr[m.body_jntadr[b]]
    u=np.array([np.cos(uang),np.sin(uang),0.]); a=np.cross([0,0,1.],u)
    Rt=R.from_rotvec(-a*np.radians(th_deg)); Rw=Rt*R.from_euler('z',psi_deg,degrees=True)
    # drink half extent along u (drink yaw ~0): 
    half=abs(u[0])*0.031+abs(u[1])*0.0205
    rim_t=np.array([dp[0],dp[1],0.0])-u*(half-ov); rim_t[2]=0.510
    p_rim=Rw.inv().apply(Rt.apply(u*0.062+np.array([0,0,0.018]))) # body-frame point that maps to u*0.062 before tilt
    p_rim=np.array([0,0,0]) # recompute simply: body-frame rim point in direction u
    pr_body=R.from_euler('z',psi_deg,degrees=True).inv().apply(u*0.062)+np.array([0,0,0.018])
    c=rim_t-Rw.apply(pr_body)
    qq=Rw.as_quat(); d.qpos[qa:qa+3]=c; d.qpos[qa+3:qa+7]=[qq[3],*qq[:3]]; d.qvel[da:da+6]=0; mujoco.mj_forward(m,d)
    if gap is not None:
        cb=body(m,"can_0"); ca=m.jnt_qposadr[m.body_jntadr[cb]]; cd=m.jnt_dofadr[m.body_jntadr[cb]]
        cp=c-u*(0.065+0.0335+gap); d.qpos[ca:ca+3]=[cp[0],cp[1],0.5155]; d.qpos[ca+3:ca+7]=[1,0,0,0]; d.qvel[cd:cd+6]=0; mujoco.mj_forward(m,d)
    lift=0
    if DBG:
        for i in range(d.ncon):
            cc=d.contact[i]; g1,g2=cc.geom1,cc.geom2
            if b in (m.geom_bodyid[g1],m.geom_bodyid[g2]) and cc.dist<-0.0005: print('  pen',mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g1),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g2),np.round(cc.pos,3),round(cc.dist,4))
    while True:
        pen=[1 for i in range(d.ncon) if b in (m.geom_bodyid[d.contact[i].geom1],m.geom_bodyid[d.contact[i].geom2]) and d.contact[i].dist<-0.0005]
        if not pen or lift>0.05: break
        d.qpos[qa+2]+=0.001; lift+=0.001; mujoco.mj_forward(m,d)
    tr=[]
    for t in range(NST//20):
        h.step(20); tr.append(round(float(np.degrees(np.arccos(np.clip(d.xmat[b][8],-1,1)))),1))
    dr=d.xpos[body(m,"boxed_drink_0")]; bb=np.round(bowlo.region_objects['bowl_0_region'][0].bbox,3)
    print(f"s{seed} u{np.degrees(uang):.0f} th{th_deg} psi{psi_deg} ov{ov:.3f} lift{lift:.3f} -> bowl {np.round(d.xpos[b],3)} tilts {tr[::3]} IN {inreg(dr)} canIN {inreg(d.xpos[body(m,'can_0')])} can {np.round(d.xpos[body(m,'can_0')],3)} bbox {bb}",flush=True)
    return h
for args in eval(sys.argv[2]): trial(*args)
