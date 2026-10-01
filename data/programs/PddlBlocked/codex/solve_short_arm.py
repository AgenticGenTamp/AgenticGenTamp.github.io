"""Empirically solve a shorter, orientation-preserving blocker grasp posture."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from env_client import make_env
from approach import GeneratedApproach


def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def pose(s):
    return (np.array([g(s,'blocker','pose_'+x) for x in 'xyz']),
            Rotation.from_quat([g(s,'blocker','pose_q'+x) for x in 'xyzw']))
def qvals(s): return np.array([g(s,'robot','joint_'+str(i+1)) for i in range(7)])


e=make_env();s,info=e.reset(seed=27);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
for _ in range(20):
    s,*_=e.step(p.get_action(s))
    if p.stage==2:break
# Lift and translate the base outward without advancing the policy to its release.
for _ in range(4):
    a=np.zeros(11,np.float32);a[:2]=.2*p.out;a[4]=-.05;s,*_=e.step(a)
assert g(s,'robot','grasp_active')>.5
base=np.array([g(s,'robot','base_x'),g(s,'robot','base_y')]); yaw=g(s,'robot','base_rot')
p0,r0=pose(s); c,z=math.cos(yaw),math.sin(yaw); R2=np.array([[c,-z],[z,c]])
local=R2.T@(p0[:2]-base)
target_p=p0.copy();target_p[:2]=base+R2@np.array([.42,local[1]])
target_r=r0
print('start q',qvals(s),'local',local,'goal',target_p)
for iteration in range(35):
    curp,curr=pose(s); err=np.r_[target_p-curp,(target_r*curr.inv()).as_rotvec()]
    if np.linalg.norm(err[:3])<.008 and np.linalg.norm(err[3:])<.04:break
    J=np.zeros((6,7)); eps=.015
    for j in range(7):
        beforep,beforer=pose(s);a=np.zeros(11,np.float32);a[3+j]=eps;s2,*_=e.step(a)
        afterp,afterr=pose(s2)
        if np.linalg.norm(afterp-beforep)>1e-5:
            J[:3,j]=(afterp-beforep)/eps
            J[3:,j]=(afterr*beforer.inv()).as_rotvec()/eps
            a[3+j]=-eps;s,*_=e.step(a)
        else:s=s2
    dq=J.T@np.linalg.solve(J@J.T+.01*np.eye(6),err)
    dq=np.clip(dq,-.12,.12);a=np.zeros(11,np.float32);a[3:10]=dq;s,*_=e.step(a)
    print(iteration,'err',np.round(err,3),'dq',np.round(dq,3))
curp,curr=pose(s);q=qvals(s);local=R2.T@(curp[:2]-base)
print('RESULT q',q.tolist(),'local_offset',local.tolist(),'z',curp[2],
      'rot_error',(curr*r0.inv()).as_rotvec().tolist())
e.close()
