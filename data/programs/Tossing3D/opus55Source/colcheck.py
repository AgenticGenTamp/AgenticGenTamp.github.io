import numpy as np, mujoco
from approach import Kin, RETRACT, QNOM, WINDUP
k=Kin(); m=k.m; d=k.d
names={i:mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)}
def contacts(base,q):
    k.fk(base,q); mujoco.mj_collision(m,d)
    out=[]
    for c in d.contact[:d.ncon]:
        b1=names[m.geom_bodyid[c.geom1]]; b2=names[m.geom_bodyid[c.geom2]]
        if ('robot' in b1 and 'robot' in b2) or ('robot' in b1+b2 and 'world' in (b1,b2)):
            if c.dist<0: out.append((b1,b2,round(c.dist,3)))
    return out
def path(base,q0,q1,n=30):
    bad=[]
    for t in np.linspace(0,1,n):
        c=contacts(base,q0+(q1-q0)*t)
        if c: bad.append((round(t,2),c[:2]))
    return bad
if __name__=='__main__':
    qa=np.array([-0.005,1.857,3.148,-2.037,-0.006,1.795,1.555])
    print('retract', contacts([0,0,0],RETRACT))
    print('qa', contacts([0,0,0],qa))
    print('path', path([0,0,0],RETRACT,qa)[:3])
    print('windup', contacts([0,0,0],WINDUP))
    from approach import tilt_rot
    # pregrasp IK for typical cube at 0.45 ahead of arm
    for hd_off in [0]:
        bt=np.array([0,0,0.]); c=np.array([0.57,0,0.025])
        Rg=tilt_rot(0,0.9)
        qa,err=k.ik(bt,QNOM,c-0.12*Rg[:,2]+[0,0,0.03],Rg,qnom=QNOM)
        print('qa',qa.round(3),err, contacts(bt,qa)); print('path', path(bt,RETRACT,qa)[:3])
        qb,err=k.ik(bt,qa,c+0.01*Rg[:,2],Rg,qnom=qa); print('qb',qb.round(3),err,contacts(bt,qb))
