import numpy as np, json
from env_client import make_env
import fk, ik

JN = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']

def robot_q(obs):
    r = obs.get_object_from_name('robot')
    return np.array([float(obs.get(r,f)) for f in JN])

def robot_base(obs):
    r = obs.get_object_from_name('robot')
    return np.array([float(obs.get(r,f)) for f in ['pos_base_x','pos_base_y','pos_base_rot']])

def rinfo(obs):
    r = obs.get_object_from_name('robot')
    return {f: round(float(obs.get(r,f)),4) for f in ['finger_state','grasp_active','grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']}

def opos(obs, name):
    o = obs.get_object_from_name(name)
    return np.array([float(obs.get(o,f)) for f in ['pose_x','pose_y','pose_z']])

def rotz_mat(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def down_R(yaw):
    # ee z axis pointing -z world, x axis rotated by yaw
    R = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
    return rotz_mat(yaw) @ R

def quat_to_mat(q):
    x,y,z,w = q
    n = np.sqrt(x*x+y*y+z*z+w*w)
    if n<1e-12: return np.eye(3)
    x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([
      [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def opose(obs,name):
    o=obs.get_object_from_name(name)
    p=np.array([float(obs.get(o,f)) for f in ['pose_x','pose_y','pose_z']])
    q=[float(obs.get(o,f)) for f in ['pose_qx','pose_qy','pose_qz','pose_qw']]
    T=np.eye(4); T[:3,:3]=quat_to_mat(q); T[:3,3]=p
    return T

def grasp_tf_mat(obs):
    r=obs.get_object_from_name('robot')
    p=np.array([float(obs.get(r,f)) for f in ['grasp_tf_x','grasp_tf_y','grasp_tf_z']])
    q=[float(obs.get(r,f)) for f in ['grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']]
    T=np.eye(4); T[:3,:3]=quat_to_mat(q); T[:3,3]=p
    return T
