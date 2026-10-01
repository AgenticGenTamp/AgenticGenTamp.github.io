import numpy as np
F=['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_arm_joint3','pos_arm_joint4','pos_arm_joint5','pos_arm_joint6','pos_arm_joint7','pos_gripper']
def rstate(obs):
    r=[o for o in obs.get_object_names() if o=='robot'][0]
    r=obs.get_object_from_name('robot')
    return np.array([obs.get(r,f) for f in F])
def cubes(obs):
    out={}
    for n in sorted(obs.get_object_names()):
        if n.startswith('cube'):
            o=obs.get_object_from_name(n)
            out[n]=np.array([obs.get(o,f) for f in ['x','y','z','qw','qx','qy','qz']])
    return out
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def drive(env, obs, target, grip, steps=200, tol=0.01, gain=1.0, log=None):
    """target: 10-vector [bx,by,bth,q1..q7]"""
    target=np.array(target,float)
    for t in range(steps):
        s=rstate(obs)
        err=target-s[:10]
        err[2]=wrap(err[2]); err[3:]=wrap(err[3:])
        if np.max(np.abs(err))<tol: return obs,t,True
        a=np.zeros(11,np.float32)
        a[:10]=np.clip(gain*err,-0.1,0.1)
        a[10]=grip
        obs,rew,te,tr,info=env.step(a)
        if log is not None: log.append((rstate(obs),rew))
    return obs,steps,False

from kin import fk_arm, ik
MX=0.1193; ZOFF=0.253   # grasped cube center (base frame) = (fk.x+MX, fk.y, fk.z+ZOFF) for down-pointing gripper
HOME=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
def Rdown(yaw):
    return np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])@np.array([[0,1,0],[1,0,0],[0,0,-1.]])
def world_to_base(p, s):
    c,sn=np.cos(s[2]),np.sin(s[2]); d=np.array(p[:2])-s[:2]
    return np.array([c*d[0]+sn*d[1], -sn*d[0]+c*d[1], p[2]])
def arm_q_for(p_base, yaw_g, q0):
    """p_base: desired grasped-cube-center in base frame. returns joint target"""
    p=np.array([p_base[0]-MX, p_base[1], p_base[2]-ZOFF])
    q,e1,e2=ik(p,Rdown(yaw_g),q0)
    return q,e1
def pick(env, obs, name, standoff=0.55, grip_wait=12):
    """Drive base so cube at (standoff,0) in base frame facing it, grasp, lift."""
    s=rstate(obs); c=cubes(obs)[name]
    d=c[:2]-s[:2]; th=np.arctan2(d[1],d[0])
    bx=c[:2]-standoff*np.array([np.cos(th),np.sin(th)])
    base=np.array([bx[0],bx[1],th])
    obs,_,_=drive(env,obs,np.concatenate([base,HOME]),0.0,steps=300,tol=0.01)
    s=rstate(obs); c=cubes(obs)[name]
    cb=world_to_base(c[:3],s)
    # cube yaw relative to base
    qw,qx,qy,qz=c[3:]; cy=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    yg=((cy-s[2]+np.pi/4)%(np.pi/2))-np.pi/4
    q=s[3:10]
    for dz,g in [(0.12,0),(0.0,0)]:
        qd,_=arm_q_for(cb+np.array([0,0,dz]),yg,q)
        obs,_,_=drive(env,obs,np.concatenate([s[:3],qd]),g,steps=200,tol=0.005)
        q=rstate(obs)[3:10]
    for _ in range(grip_wait):
        a=np.zeros(11,np.float32); a[3:10]=0; a[10]=1; obs,*_=env.step(a)
    qd,_=arm_q_for(cb+np.array([0,0,0.15]),yg,q)
    obs,_,_=drive(env,obs,np.concatenate([s[:3],qd]),1.0,steps=200,tol=0.005)
    return obs
