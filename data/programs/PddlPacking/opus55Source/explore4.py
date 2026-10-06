import numpy as np, time
from simcore import *
import pybullet as p
s = Sim(4)
st = s.get_state()
q0 = np.array(st[:10], float)
cid=s.cid; r=s.robot
mov=[j for j in range(p.getNumJoints(r,physicsClientId=cid)) if p.getJointInfo(r,j,physicsClientId=cid)[2]!=p.JOINT_FIXED]
idx=[mov.index(j) for j in s.joints]
print(len(mov), idx)
s.set_conf(q0)
t=time.time()
for i in range(100):
    js=[x[0] for x in p.getJointStates(r,mov,physicsClientId=cid)]
    ls=p.getLinkState(r,s.tool,computeForwardKinematics=1,physicsClientId=cid)
    lt,ang=p.calculateJacobian(r,s.tool,(0,0,0),js,[0.0]*len(js),[0.0]*len(js),physicsClientId=cid)
print('jac time',(time.time()-t)/100)
lt=np.array(lt)[:,idx]; ang=np.array(ang)[:,idx]
# numeric compare
cp,cq=s.fk(q0)
for c in range(10):
    q2=q0.copy(); q2[c]+=1e-5
    p2,qq=s.fk(q2)
    print(c, np.round((p2-cp)/1e-5,3), np.round(lt[:,c],3), np.round(rotvec_err(qq,cq)/1e-5,3), np.round(ang[:,c],3))
