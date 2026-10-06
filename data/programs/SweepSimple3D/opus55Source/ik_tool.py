import numpy as np, mujoco
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:SweepSimple3DEnv",
    count_kwarg="num_objects", count_object_prefix="cube_", constant_object_env_kwargs={'scene_bg': False},
    design_counts=[1,5,10,50], eval_counts=[1,5,10,50])
s,_=env.reset(seed=0, options={'object_count':1})
oc = env._current_backend._object_centric_env
m = oc._robot_env.sim.model.mj_model; d0 = oc._robot_env.sim.data.mj_data
d = mujoco.MjData(m); d.qpos[:] = d0.qpos
J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
bx,by,bth = J('robot_joint_x'),J('robot_joint_y'),J('robot_joint_th')
arm = [J('robot_joint_%d'%i) for i in range(1,8)]
armdof = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,'robot_joint_%d'%i)] for i in range(1,8)]
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'robot_pinch_site')
d.qpos[bx]=0; d.qpos[by]=0; d.qpos[bth]=0
def fk(q):
    d.qpos[arm]=q; mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
    return d.site_xpos[site].copy(), d.site_xmat[site].reshape(3,3).copy()
def ik(p, R, q0, iters=300):
    q=np.array(q0,float)
    for it in range(iters):
        x, Rc = fk(q)
        ep = p-x
        er = 0.5*(np.cross(Rc[:,0],R[:,0])+np.cross(Rc[:,1],R[:,1])+np.cross(Rc[:,2],R[:,2]))
        if np.linalg.norm(ep)<1e-5 and np.linalg.norm(er)<1e-4: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        Jm=np.vstack([jp[:,armdof],jr[:,armdof]]); e=np.concatenate([ep,er])
        dq=Jm.T@np.linalg.solve(Jm@Jm.T+1e-4*np.eye(6),e)
        q+=np.clip(dq,-0.2,0.2)
    return q, np.linalg.norm(ep), np.linalg.norm(er)
q_init = d0.qpos[arm].copy()
if __name__=='__main__':
    x,R=fk(q_init); print('init site base-frame',x); print(R.round(3))
    # base-frame origin of arm mount
    print('arm base body', d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,'robot_gen3/base_link')])
