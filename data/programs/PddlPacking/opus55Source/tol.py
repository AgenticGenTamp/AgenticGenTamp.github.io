import numpy as np
exec(open('approach.py').read())
s=Sim(1); st=s.get_state(); P=Planner(s)
# put block at (0,-0.1), yaw 0
st=st.copy(); o=39; st[o:o+3]=[-0.1,0.0,0.781]; st[o+3:o+7]=[0,0,0,1]
s.set_state(st)
qc=np.array(st[:10]); qc[0]=-0.6
c=np.array([-0.1,0.0,0.781])
def test(dx,dy,dz,dyaw,tilt=0.0):
    quat=qmul(yaw_quat(dyaw),Q0)
    if tilt: quat=qmul(p.getQuaternionFromEuler([tilt,0,0]),quat)
    q=s.ik(c+np.array([dx,dy,dz]),quat,qc,list(range(3,10)))
    if q is None: return 'I'
    s.set_state(st); s.set_conf(q)
    with s.env.client():
        from pybullet_tools.utils import pairwise_collision
        if pairwise_collision(s.robot,s.env.blocks[0]): return 'C'
        if pairwise_collision(s.robot,s.env.table): return 'T'
        g=s.env._grasp_candidate()
    return 'G' if g is not None else 'n'
print('yaw sweep (dz=0.03):', ''.join(test(0,0,0.03,np.radians(a)) for a in range(0,50,3)))
print('dz sweep -0.02..0.08:', ''.join(test(0,0,z,0) for z in np.arange(-0.02,0.081,0.01)))
print('dx sweep:', ''.join(test(x,0,0.03,0) for x in np.arange(-0.04,0.041,0.005)))
print('dy sweep:', ''.join(test(0,y,0.03,0) for y in np.arange(-0.04,0.041,0.005)))
print('tilt sweep:', ''.join(test(0,0,0.03,0,np.radians(a)) for a in range(-30,31,5)))
s.set_conf(qc); 
print('gripper', [p.getJointState(s.robot,j,physicsClientId=s.cid)[0] for j in s.env._gripper_joints])
