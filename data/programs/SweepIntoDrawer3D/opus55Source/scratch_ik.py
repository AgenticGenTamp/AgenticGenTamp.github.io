import numpy as np, mujoco, time
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model
d=mujoco.MjData(m)
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
def fk(base,q):
    d.qpos[52:55]=base; d.qpos[55:62]=q; mujoco.mj_kinematics(m,d)
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3,3).copy()
lim=np.array([[-9,9],[-2.2,2.2],[-9,9],[-2.5,2.5],[-9,9],[-2.05,2.05],[-9,9]])
def ik(base,q0,pt,Rt,iters=100):
    q=q0.copy(); J=np.zeros((6,m.nv)); jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
    for i in range(iters):
        p,R=fk(base,q)
        ep=pt-p
        Re=Rt@R.T; er=0.5*(np.cross(R[:,0],Rt[:,0])+np.cross(R[:,1],Rt[:,1])+np.cross(R[:,2],Rt[:,2]))
        if np.linalg.norm(ep)<1e-4 and np.linalg.norm(er)<1e-3: break
        mujoco.mj_comPos(m,d); mujoco.mj_jacSite(m,d,jp,jr,sid)
        Jm=np.vstack([jp[:,49:56],0.5*jr[:,49:56]]); e=np.concatenate([ep,0.5*er])
        dq=Jm.T@np.linalg.solve(Jm@Jm.T+1e-3*np.eye(6),e)
        dq=np.clip(dq,-0.3,0.3); q=np.clip(q+dq,lim[:,0],lim[:,1])
    return q, np.linalg.norm(ep), np.linalg.norm(er), i
q0=obs[128:135]
print(fk(obs[125:128],q0))
# horizontal grasp: site z=(-1,0,0), site x=(0,0,1)
Rh=np.column_stack([[0,0,1],np.cross([-1,0,0],[0,0,1]),[-1,0,0]])
Rh=np.column_stack([[0,0,1],[0,1,0],[-1,0,0]])
print(np.linalg.det(Rh))
for bx in [1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6]:
    t=time.time()
    q,e1,e2,it=ik(np.array([bx,0,np.pi]),q0,np.array([0.932,0,0.325]),Rh)
    print(bx, round(e1,4),round(e2,4),it, q.round(2), round(time.time()-t,4))
qh=np.array([0,0.26,np.pi,-2.27,0,0.96,np.pi/2])
print('home fk',fk(np.array([1.5,0,np.pi]),qh))
for bx in [1.3,1.35,1.4,1.45,1.5,1.55]:
    q,e1,e2,it=ik(np.array([bx,0,np.pi]),qh,np.array([0.932,0,0.325]),Rh)
    print(bx, round(e1,4),round(e2,4),it, q.round(2))
Rd=np.column_stack([[0,1,0],[1,0,0],[0,0,-1]])  # pointing down, fingers along y
for p in [(0.6,-0.1,0.5),(0.75,-0.35,0.49),(1.0,-0.35,0.49),(1.0,-0.1,0.5)]:
    q,e1,e2,it=ik(np.array([1.5,0,np.pi]),qh,np.array(p),Rd)
    print(p, round(e1,4),round(e2,4),it, q.round(2))
