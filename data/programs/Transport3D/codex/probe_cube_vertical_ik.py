"""Try vertical-tool IK solutions at the cube's calibrated height."""
import sys, math
import numpy as np
from scipy.optimize import least_squares
from env_client import make_env
from probe_grasp_structured import command, val, fk, JOINT_OFFSET

idx=int(sys.argv[1]); rng=np.random.default_rng(idx)
lo=np.array([0.,-.35,-math.pi,-2.5,0.,-.87,math.pi/2]);hi=np.array([5.2,2.06,-1.0836,.16,5.2,1.36,8.3408])
# The known box contact has DH height .535; cube center is 75 mm lower.
r=.18+.09*(idx%5); target=np.array([r,0.,.46023324]); down=-1. if (idx//5)%2==0 else 1.
def residual(q):
 t=fk(q+JOINT_OFFSET);return np.r_[8*(t[:3,3]-target),2*(t[:3,2]-[0,0,down])]
sol=least_squares(residual,rng.uniform(lo,hi),bounds=(lo,hi),max_nfev=600)
q=sol.x
qs=[np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593]),np.array([4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916]),np.array([3.870779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434])]
offs=[(-.799987478,-.347800459,2.104792961),(.126212298,-.083534270,-3.139703362),(-.428185141,.054943428,2.978131848)]
e=make_env();s,_=e.reset(seed=1);tx=val(s,'cube0','pose_x');ty=val(s,'cube0','pose_y')
for qq,o in zip(qs,offs):s=command(e,s,[tx+o[0],ty+o[1],o[2]],qq,1.,35);s=command(e,s,[tx+o[0],ty+o[1],o[2]],qq,-1.,2)
b=np.array([tx+offs[-1][0],ty+offs[-1][1]]);s=command(e,s,[b[0],b[1],offs[-1][2]],q,1.,35)
for iy,dy in enumerate(np.arange(-.4,.401,.02)):
 xs=np.arange(-.4,.401,.02)
 if iy%2:xs=xs[::-1]
 for dx in xs:
  bb=[b[0]+dx,b[1]+dy,offs[-1][2]];s=command(e,s,bb,q,1.,3);s=command(e,s,bb,q,-1.,2)
  if val(s,'robot','grasp_active')>.5:print('HIT idx',idx,'err',np.linalg.norm(sol.fun),'xy',dx,dy,'q',q.tolist(),'off',val(s,'robot','pos_base_x')-tx,val(s,'robot','pos_base_y')-ty);e.close();raise SystemExit
print('MISS',idx,'err',np.linalg.norm(sol.fun),'q',q.tolist());e.close()
