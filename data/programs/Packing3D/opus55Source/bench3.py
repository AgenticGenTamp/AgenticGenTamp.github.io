import time, numpy as np, pybullet as p
from planner import TwinPlanner, _ik_meta
tp = TwinPlanner(2); arm = tp.arm; cid=arm.physics_client_id; rid=arm.robot_id
free=_ik_meta(arm)[0]; q=[s[0] for s in p.getJointStates(rid, free, physicsClientId=cid)]; z=[0.0]*len(free)
def tm(f,n=200):
    t=time.time()
    for _ in range(n): f()
    return round((time.time()-t)/n*1000,4)
print("jac", tm(lambda: p.calculateJacobian(rid, arm.end_effector_id, [0,0,0], q, z, z, physicsClientId=cid)))
print("reset7", tm(lambda: [p.resetJointState(rid, j, q[i], physicsClientId=cid) for i,j in enumerate(free[:7])]))
print("linkstate", tm(lambda: p.getLinkState(rid, arm.end_effector_id, computeForwardKinematics=True, physicsClientId=cid)))
print("states", tm(lambda: p.getJointStates(rid, free, physicsClientId=cid)))
A=np.random.rand(6,7)
print("solve", tm(lambda: A.T@np.linalg.solve(A@A.T+1e-3*np.eye(6), np.ones(6))))
