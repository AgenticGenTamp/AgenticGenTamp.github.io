import numpy as np, sys, mujoco
from approach import Kin, down_R, axis_R, ARM_LO, ARM_HI
K=Kin()
def ik_ns(base,q,pos,R,iters=200,qmid=None,kns=0.1):
    q=q.copy()
    for _ in range(iters):
        p,Rc=K.fk(base,q)
        er=0.5*(np.cross(Rc[:,0],R[:,0])+np.cross(Rc[:,1],R[:,1])+np.cross(Rc[:,2],R[:,2]))
        e=np.concatenate([pos-p,0.3*er])
        mujoco.mj_comPos(K.m,K.d); mujoco.mj_jacSite(K.m,K.d,K.jacp,K.jacr,K.sid)
        J=np.vstack([K.jacp[:,K.arm_v],0.3*K.jacr[:,K.arm_v]])
        Jp=J.T@np.linalg.inv(J@J.T+1e-3*np.eye(6))
        dq=Jp@e
        if qmid is not None:
            g=np.zeros(7)
            for j in [1,3,5]: g[j]=(qmid[j]-q[j])
            dq+= (np.eye(7)-Jp@J)@(kns*g)
        n=np.linalg.norm(dq)
        if n>0.3: dq*=0.3/n
        q=np.clip(q+dq,ARM_LO,ARM_HI)
    return q,np.linalg.norm(e)
def torque(base,q,point,W):
    K.fk(base,q); mujoco.mj_comPos(K.m,K.d)
    jp=np.zeros((3,K.m.nv)); jr=np.zeros((3,K.m.nv))
    bid=mujoco.mj_name2id(K.m,mujoco.mjtObj.mjOBJ_BODY,'robot_bracelet_link')
    mujoco.mj_jac(K.m,K.d,jp,jr,point,bid)
    return jp[:,K.arm_v].T@np.array([0,0,-W])
q0=np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
R0=down_R(np.pi/2); Gp=np.array([0,-0.14,0.021]); Gr=R0
c1=np.array([0.5,0.2,float(sys.argv[1]) if len(sys.argv)>1 else 0.72]); BX=float(sys.argv[2]) if len(sys.argv)>2 else -0.16
qmid=np.zeros(7)
q,_=ik_ns(np.array([BX,-0.44,0]), q0, np.array([0.5,-0.34,0.515]), R0, qmid=qmid)
print('grasp q', np.round(q,2), 'tau', np.round(torque(np.array([BX,-0.44,0]),q,np.array([0.5,-0.2,0.5]),20),1))
for phi in np.linspace(0,-np.radians(165),12):
    Rb=axis_R(np.array([1.,0,0]),phi)
    pos=c1+Rb@Gp; R=Rb@Gr
    base=np.array([BX, pos[1]-0.1, 0])
    q,err=ik_ns(base,q,pos,R,qmid=qmid)
    print(round(np.degrees(phi)), 'err',round(err,4),'q',np.round(q,2),'tau20',np.round(torque(base,q,c1,20),1))
