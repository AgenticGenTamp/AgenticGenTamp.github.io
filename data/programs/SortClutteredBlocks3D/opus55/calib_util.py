from env_client import make_env
import numpy as np
from kin import fk_arm, ik
JF=[f'pos_arm_joint{i}' for i in range(1,8)]
def rstate(obs):
    r=obs.get_object_from_name('robot')
    b=np.array([obs.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    q=np.array([obs.get(r,f) for f in JF]); return b,q
def objpos(obs,n):
    o=obs.get_object_from_name(n); return np.array([obs.get(o,f) for f in ['x','y','z']])
def act(base=(0,0,0), dq=None, grip=0.0):
    a=np.zeros(11); a[:3]=base
    if dq is not None: a[3:10]=dq
    a[10]=grip; return a
def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi
def joint_ctrl(env, obs, qt, K=3.0, steps=100, grip=0.0, tol=0.002, base=(0,0,0), verbose=False):
    for k in range(steps):
        b,q=rstate(obs); e=qt-q
        if np.max(np.abs(e))<tol: return obs,k
        obs,_,_,_,_=env.step(act(base,np.clip(K*e,-0.1,0.1),grip))
    return obs,steps
Q_HOME=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
_, R_HOME, _, _ = fk_arm(Q_HOME, 0)
def down_R(yaw=0.0):
    # bracelet frame pointing down (z up), x axis rotated by yaw in arm frame; yaw=0 ~ home orientation
    x0=R_HOME[:,0].copy(); x0[2]=0; x0/=np.linalg.norm(x0)
    z=np.array([0,0,1.]); 
    c,s=np.cos(yaw),np.sin(yaw); x=c*x0+s*np.cross(z,x0)
    y=np.cross(z,x); return np.stack([x,y,z],1)
def drive_base(env, obs, target, steps=200, grip=0.0, tol=0.002, dq_hold=None):
    for k in range(steps):
        b,q=rstate(obs); e=np.array(target)-b; e[2]=wrap(e[2])
        if np.max(np.abs(e))<tol: return obs,k
        obs,_,_,_,_=env.step(act(np.clip(e/0.87,-0.1,0.1),None,grip))
    return obs,steps
