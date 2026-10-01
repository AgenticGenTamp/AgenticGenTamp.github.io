import numpy as np
import robot
JN=['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']
def robot_q(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in JN])
def base(o):
    r=o.get_object_from_name('robot'); return np.array([float(o.get(r,f)) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
def rinfo(o):
    r=o.get_object_from_name('robot'); return {f:round(float(o.get(r,f)),4) for f in ['finger_state','grasp_active','grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']}
def opos(o,n):
    x=o.get_object_from_name(n); return np.array([float(o.get(x,f)) for f in ['pose_x','pose_y','pose_z']])
def half(o,n):
    x=o.get_object_from_name(n); return np.array([float(o.get(x,f)) for f in ['half_extent_x','half_extent_y','half_extent_z']])
def grasp_tf(o):
    r=o.get_object_from_name('robot')
    p=np.array([float(o.get(r,f)) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z']])
    q=[float(o.get(r,f)) for f in ['grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']]
    T=np.eye(4); T[:3,:3]=robot.quat_to_mat(*q); T[:3,3]=p; return T
def go_world(env,obs,p_world,R,nmax=60,tol=1e-4):
    """servo gripper to world pose"""
    q=robot_q(obs); b=base(obs)
    B=robot.base_tf(*b); Binv=np.linalg.inv(B)
    pl = (Binv @ np.append(p_world,1.0))[:3]
    Rl = B[:3,:3].T @ R
    blocked=False
    for i in range(nmax):
        dq,ep,ew = robot.ik_step(q,pl,Rl,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf=env.step(a)
        qn=robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-6:
            blocked=True; q=qn; break
        q=qn
        if ep<tol and ew<1e-3: break
    T=robot.base_tf(*base(obs)) @ robot.fk_tool(q)
    return obs, blocked, np.linalg.norm(T[:3,3]-p_world)
