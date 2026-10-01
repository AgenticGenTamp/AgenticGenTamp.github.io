from env_client import make_env
import numpy as np
from kin import fk_arm, ik
J=['joint_%d'%i for i in range(1,8)]
Rdown=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
RF=['pos_base_x','pos_base_y','pos_base_rot','joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7','finger_state','grasp_active','grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']
OF=['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw','grasp_active']
class E:
    def __init__(s, seed):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed)
    def rob(s):
        r=s.obs.get_object_from_name('robot'); return {f:float(s.obs.get(r,f)) for f in RF}
    def o(s,name):
        r=s.obs.get_object_from_name(name); return {f:float(s.obs.get(r,f)) for f in OF}
    def q(s):
        r=s.rob(); return np.array([r[j] for j in J])
    def step(s,a):
        s.obs,s.rew,s.term,s.trunc,_=s.env.step(np.array(a,dtype=np.float32)); return s.obs
    def gripper(s,v):
        a=np.zeros(11); a[10]=v; s.step(a)
    def goto(s,qt,maxstep=0.1):
        for _ in range(300):
            qc=s.q(); d=qt-qc
            if np.max(np.abs(d))<1e-4: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-maxstep,maxstep); s.step(a)
            if np.max(np.abs(s.q()-qc))<1e-7: return False
        return False
    def base_to(s,x,y,rot=None):
        for _ in range(300):
            r=s.rob(); d=np.array([x-r['pos_base_x'],y-r['pos_base_y'],0 if rot is None else rot-r['pos_base_rot']])
            if np.max(np.abs(d))<1e-5: return True
            a=np.zeros(11); a[:3]=np.clip(d,-0.2,0.2); s.step(a)
            r2=s.rob()
            if abs(r2['pos_base_x']-r['pos_base_x'])+abs(r2['pos_base_y']-r['pos_base_y'])+abs(r2['pos_base_rot']-r['pos_base_rot'])<1e-9: return False
        return False
    def ee_to(s,p,R=Rdown,maxstep=0.05):
        qt,ep,er=ik(np.array(p),R,s.q()); return s.goto(qt,maxstep), ep
M_OFF=0.12; H=0.2748
def place_and_reach(e, wx, wy, wz, brot=0.0, yaw=0.0, reach=0.5, pre=0.15):
    from kin import rpy
    R=rpy(0,0,yaw)@Rdown
    ax,ay=wx-reach*np.cos(brot),wy-reach*np.sin(brot)
    bx,by=ax-M_OFF*np.cos(brot),ay-M_OFF*np.sin(brot)
    okb=e.base_to(bx,by,brot)
    e.ee_to([reach,0,wz-H+pre],R); ok,_=e.ee_to([reach,0,wz-H],R,maxstep=0.02)
    return okb,ok
def lin_to(e, p, R=Rdown, seg=0.01):
    p=np.array(p,float); cur=fk_arm(e.q())[:3,3]
    n=max(1,int(np.ceil(np.linalg.norm(p-cur)/seg)))
    for i in range(1,n+1):
        qt,_,_=ik(cur+(p-cur)*i/n,R,e.q())
        if not e.goto(qt,0.2): return False
    return True
def place_and_reach2(e, wx, wy, wz, brot=0.0, yaw=0.0, reach=0.5, pre=0.15):
    from kin import rpy
    R=rpy(0,0,yaw)@Rdown
    ax,ay=wx-reach*np.cos(brot),wy-reach*np.sin(brot)
    bx,by=ax-M_OFF*np.cos(brot),ay-M_OFF*np.sin(brot)
    okb=e.base_to(bx,by,brot)
    e.ee_to([reach,0,wz-H+pre],R); ok=lin_to(e,[reach,0,wz-H],R)
    return okb,ok
