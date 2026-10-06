import sys, math, numpy as np
sys.path.insert(0,'.')
from approach import *
from approach import _rot_err
import pybullet as p
sim = Sim(0); e=sim.env
with e.client():
    from pybullet_tools.utils import get_movable_joints
    mj = get_movable_joints(e.robot)
    cid = e.client_id
    print(cid, e.client_id)
    q = sim.get_conf()+0.1
    sim.set_conf(q)
    js = [p.getJointState(e.robot, j, physicsClientId=cid)[0] for j in mj]
    lin, ang = p.calculateJacobian(e.robot, e.tool_link, [0,0,0], js, [0.0]*len(js), [0.0]*len(js), physicsClientId=cid)
    lin=np.array(lin); ang=np.array(ang)
    idx = [mj.index(j) for j in list(e.base_joints)+list(e.arm_joints)]
    print(idx)
    p0,R0 = sim.tool(q)
    for i in range(10):
        dq=q.copy(); dq[i]+=1e-5
        p2,R2=sim.tool(dq)
        print(i, np.round((p2-p0)/1e-5,3), np.round(lin[:,idx[i]],3), np.round(_rot_err(R2,R0)/1e-5,3), np.round(ang[:,idx[i]],3))
    print(p.getLinkState(e.robot, e.tool_link, physicsClientId=cid)[2])
