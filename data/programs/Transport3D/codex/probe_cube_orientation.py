"""Cube grasp search emphasizing tool orientation rather than box-pose offsets."""
import math
import sys
import numpy as np
from scipy.optimize import least_squares
from env_client import make_env
from probe_grasp_structured import command, val, fk, JOINT_OFFSET

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
lo=np.array([0.,-.35,-math.pi,-2.5,0.,-.87,math.pi/2])
hi=np.array([5.2,2.06,-1.0836,.16,5.2,1.36,8.3408])

# Target the cube-height shell.  Orient the three tool axes in all cardinal
# directions; unlike earlier probes this does not inherit the box wrist pose.
axes=[]
card=np.eye(3)
for closing_axis in range(3):
  for sign in (-1.,1.):
    for roll in range(4):
      z=card[:,closing_axis]*sign
      x=card[:,(closing_axis+1)%3]
      # rotate the secondary constraint around the primary axis
      x=np.roll(x,roll%3) if roll < 3 else -x
      if abs(np.dot(x,z))>.1: x=card[:,(closing_axis+2)%3]
      axes.append((z,x))

rng=np.random.default_rng(1000+batch)
candidates=[]
for ci in range(24):
  zaxis,xaxis=axes[(batch*24+ci)%len(axes)]
  radius=(.18,.28,.38,.48)[ci%4]
  height=(.40,.46,.52)[(ci//4)%3]
  target=np.array([radius,0.,height])
  def residual(q):
    t=fk(q+JOINT_OFFSET)
    return np.r_[10*(t[:3,3]-target), 2.5*(t[:3,2]-zaxis), 1.2*(t[:3,0]-xaxis)]
  sol=least_squares(residual,rng.uniform(lo,hi),bounds=(lo,hi),max_nfev=500)
  if np.linalg.norm(sol.fun)<1.0:
    candidates.append((np.linalg.norm(sol.fun),sol.x,zaxis,xaxis,radius,height))
print('batch',batch,'ik',len(candidates),flush=True)

# A fresh episode per posture.  Sweep local base offsets fairly coarsely; a
# successful gripper contact region should span multiple centimeters.
for ci,item in enumerate(candidates):
  err,q,za,xa,radius,height=item
  env=make_env();s,_=env.reset(seed=seed)
  tx,ty=val(s,'cube0','pose_x'),val(s,'cube0','pose_y')
  # Known collision-clearing winding only; final posture is independently IKed.
  qs=[np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593]),
      np.array([4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916])]
  os=[(-.80,-.348,2.105),(.126,-.084,-3.140)]
  for qq,o in zip(qs,os): s=command(env,s,[tx+o[0],ty+o[1],o[2]],qq,1.,35)
  yaw=(-math.pi,-math.pi/2,0,math.pi/2)[ci%4]
  # Use FK radial reach as initial horizontal translation, then compact scan.
  for iy,dy in enumerate(np.linspace(-.28,.28,8)):
    xs=np.linspace(-.65,.15,11)
    if iy%2: xs=xs[::-1]
    for dx in xs:
      base=[tx+dx,ty+dy,yaw]
      s=command(env,s,base,q,1.,10)
      s=command(env,s,base,q,-1.,2)
      if val(s,'robot','grasp_active')>.5:
        rq=[val(s,'robot','joint_%d'%j) for j in range(1,8)]
        rb=[val(s,'robot',f) for f in ('pos_base_x','pos_base_y','pos_base_rot')]
        print('CUBE_ORIENTATION_HIT',{'batch':batch,'candidate':ci,'err':err,
          'q':rq,'base_offset':[rb[0]-tx,rb[1]-ty,rb[2]],
          'zaxis':za.tolist(),'xaxis':xa.tolist(),'radius':radius,'height':height,
          'tf':[val(s,'robot','grasp_tf_'+c) for c in 'xyz']},flush=True)
        env.close();raise SystemExit
  print('miss',ci,'err',round(err,3),'za',za.tolist(),'h',height,
        'q',np.round(q,6).tolist(),flush=True)
  env.close()
print('NONE',batch,flush=True)
