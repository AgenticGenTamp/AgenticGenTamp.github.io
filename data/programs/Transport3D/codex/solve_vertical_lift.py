"""Estimate attached-object Jacobian online and test a vertical Cartesian lift."""
import math, numpy as np
from scipy.spatial.transform import Rotation
from env_client import make_env
from probe_grasp_structured import command,val
from replay_grasp_route import route

def pose(s,n):
 p=np.array([val(s,n,'pose_'+c) for c in 'xyz'])
 q=np.array([val(s,n,'pose_q'+c) for c in 'xyzw'])
 return p,q
def step(e,s,a):return e.step(np.asarray(a,np.float32))[0]
e=make_env();s,_=e.reset(seed=0,options={'object_count':0});tx,ty=val(s,'box0','pose_x'),val(s,'box0','pose_y')
for idx in [24,26,27]:
 q,r,ang,rot=route[idx];b=[tx+r*math.cos(ang),ty+r*math.sin(ang),rot]
 s=command(e,s,b,q,1.,35);s=command(e,s,b,q,-1.,2)
print('held',val(s,'robot','grasp_active'),pose(s,'box0'))
path=[]
for it in range(18):
 p0,q0=pose(s,'box0');J=np.zeros((6,7));eps=.025
 for j in range(7):
  a=np.zeros(11);a[3+j]=eps;a[10]=-1;sp=step(e,s,a);pp,qp=pose(sp,'box0')
  J[:3,j]=(pp-p0)/eps
  J[3:,j]=(Rotation.from_quat(qp)*Rotation.from_quat(q0).inv()).as_rotvec()/eps
  a[3+j]=-eps;s=step(e,sp,a)
 twist=np.r_[0,0,.06,-2*Rotation.from_quat(q0).as_rotvec()]
 dq=J.T@np.linalg.solve(J@J.T+.01*np.eye(6),twist)
 dq=np.clip(dq,-.08,.08);a=np.zeros(11);a[3:10]=dq;a[10]=-1;s=step(e,s,a);path.append(dq)
 print('up',it,np.round(pose(s,'box0')[0],3),np.round(pose(s,'box0')[1],3),np.round(dq,3))
# translate attached object over table
for _ in range(20):
 p,_=pose(s,'box0');a=np.zeros(11);a[:2]=np.clip([.55-p[0],-p[1]],-.2,.2);a[10]=-1;s=step(e,s,a)
# Remove residual tilt while safely above the table.
for it in range(8):
 p0,q0=pose(s,'box0');rv=Rotation.from_quat(q0).as_rotvec()
 if np.linalg.norm(rv)<.002: break
 J=np.zeros((6,7));eps=.025
 for j in range(7):
  a=np.zeros(11);a[3+j]=eps;a[10]=-1;sp=step(e,s,a);pp,qp=pose(sp,'box0')
  J[:3,j]=(pp-p0)/eps;J[3:,j]=(Rotation.from_quat(qp)*Rotation.from_quat(q0).inv()).as_rotvec()/eps
  a[3+j]=-eps;s=step(e,sp,a)
 twist=np.r_[0,0,0,-rv];dq=J.T@np.linalg.solve(J@J.T+.003*np.eye(6),twist);dq=np.clip(dq,-.04,.04)
 a=np.zeros(11);a[3:10]=dq;a[10]=-1;s=step(e,s,a)
 print('level',np.round(pose(s,'box0')[0],3),np.round(pose(s,'box0')[1],4))
# Recompute the local Jacobian while descending; this retains an upright box.
for it in range(20):
 p0,q0=pose(s,'box0');J=np.zeros((6,7));eps=.025
 if p0[2] <= .501: break
 for j in range(7):
  a=np.zeros(11);a[3+j]=eps;a[10]=-1;sp=step(e,s,a);pp,qp=pose(sp,'box0')
  J[:3,j]=(pp-p0)/eps;J[3:,j]=(Rotation.from_quat(qp)*Rotation.from_quat(q0).inv()).as_rotvec()/eps
  a[3+j]=-eps;s=step(e,sp,a)
 twist=np.r_[0,0,-min(.05,p0[2]-.5),-2*Rotation.from_quat(q0).as_rotvec()]
 dq=J.T@np.linalg.solve(J@J.T+.01*np.eye(6),twist);dq=np.clip(dq,-.08,.08)
 p,_=pose(s,'box0');a=np.zeros(11);a[:2]=np.clip([.55-p[0],-p[1]],-.2,.2);a[3:10]=dq;a[10]=-1;s=step(e,s,a)
 print('down',np.round(pose(s,'box0')[0],3),np.round(pose(s,'box0')[1],3))
a=np.zeros(11);a[10]=1
for k in range(4):s,r,t,tr,i=e.step(a);print('release',k,r,t,np.round(pose(s,'box0')[0],3),np.round(pose(s,'box0')[1],3))
e.close()
