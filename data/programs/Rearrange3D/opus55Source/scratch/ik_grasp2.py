import time, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
oc = env._kinder_env._object_centric_env
env.reset(seed=0)
re = oc._robot_env; m, d = re.sim.model.mj_model, re.sim.data.mj_data
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
arm_q = [m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot_joint_{i}")] for i in range(1,8)]
arm_v = [m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot_joint_{i}")] for i in range(1,8)]
def ik(pos, q0, iters=200):
    dd = mujoco.MjData(m); dd.qpos[:] = d.qpos; q=q0.copy()
    for _ in range(iters):
        dd.qpos[arm_q]=q; mujoco.mj_kinematics(m,dd); mujoco.mj_comPos(m,dd)
        p=dd.site_xpos[sid]; R=dd.site_xmat[sid].reshape(3,3)
        ep=pos-p; z=R[:,2]; er=np.cross(z,np.array([0,0,-1.]))  # keep z-axis down
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,dd,jp,jr,sid)
        J=np.vstack([jp[:,arm_v],jr[:,arm_v]]); e=np.concatenate([ep,0.5*er])
        q+=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(6),e)
        if np.linalg.norm(ep)<1e-3: break
    return q, np.linalg.norm(ep)
def cur(): return np.array(d.qpos[arm_q]).copy()
def objpos(n): b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n); return d.xpos[b].copy()
def goto(qt, grip, n=100, tol=0.01):
    for k in range(n):
        a=np.zeros(11); a[3:10]=np.clip(qt-cur(),-0.1,0.1); a[10]=grip; env.step(a)
        if np.abs(qt-cur()).max()<tol and k>2: break
    return k+1
for _ in range(15):
    b=np.array(re.qpos["base"]); a=np.zeros(11); a[:3]=np.clip(np.array([-0.05,-0.25,0])-b,-0.1,0.1); env.step(a)
print("base now",np.round(np.array(re.qpos["base"]),3))
can=objpos("can_0"); print("can",np.round(can,3),"pinch",np.round(d.site_xpos[sid],3))
t=time.time(); q1,e=ik(can+[0,0,0.12],cur()); print("ik err",e,"time %.3f"%(time.time()-t))
print("steps pre",goto(q1,0,tol=0.005)); print("pinch",np.round(d.site_xpos[sid],3))
q2,e=ik(can+[0,0,0.06],q1); goto(q2,0,tol=0.005); q2,e=ik(can+[0,0,0.0],q2); print("steps down",goto(q2,0,tol=0.005),"pinch",np.round(d.site_xpos[sid],3))
for _ in range(8): a=np.zeros(11); a[3:10]=np.clip(q2-cur(),-.1,.1); a[10]=1; env.step(a)
print("grip state", round(float(oc._get_current_state().get(oc._get_current_state().get_object_from_name('robot'),'pos_gripper')),3))
q3,e=ik(can+[0,0,0.15],q2); print("steps lift",goto(q3,1)); print("can after lift",np.round(objpos("can_0"),3))
bowl=objpos("bowl_0"); tgt=bowl+[0,-0.11,0]; tgt[2]=can[2]
q4,e=ik(tgt+[0,0,0.15],q3); print("steps move",goto(q4,1)); print("can",np.round(objpos("can_0"),3))
q5,e=ik(tgt+[0,0,0.01],q4); print("steps lower",goto(q5,1))
for _ in range(8): a=np.zeros(11); a[3:10]=np.clip(q5-cur(),-.1,.1); a[10]=0; obs,r,term,_,_=env.step(a)
q6,e=ik(tgt+[0,0,0.15],q5); goto(q6,0); obs,r,term,_,_=env.step(np.zeros(11))
print("can final",np.round(objpos("can_0"),3),"bowl",np.round(objpos("bowl_0"),3),"term",term,"timestep",re.timestep)
