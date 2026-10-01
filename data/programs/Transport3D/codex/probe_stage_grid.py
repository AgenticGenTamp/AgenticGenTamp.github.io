"""Grid-search box-tool staging paths on one-cube episodes."""
import sys
import numpy as np
from env_client import make_env

Q=np.array([[.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593],[4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916],[4.170779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434]])
OFF=np.array([[-.799987478,-.347800459,2.104792961],[.126212298,-.083534270,-3.139703362],[-.428185141,.054943428,2.978131848]])
def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
def xy(s,n): return np.array([g(s,n,'pose_x'),g(s,n,'pose_y')])
def step_goal(e,s,b,q,grip,n):
 for _ in range(n):
  a=np.zeros(11,np.float32); cur=np.array([g(s,'robot','pos_base_x'),g(s,'robot','pos_base_y'),g(s,'robot','pos_base_rot')]); cq=np.array([g(s,'robot','joint_%d'%i) for i in range(1,8)])
  a[:3]=np.clip(b-cur,-.2,.2);a[3:10]=np.clip(q-cq,-.2,.2);a[10]=grip;s,r,t,tr,_=e.step(a)
 return s
def run(seed,dist,lat,route,lift,ls=10,carry=35):
 e=make_env();s,_=e.reset(seed=seed,options={'object_count':1}); tx,ty=xy(s,'box0')
 for q,o in zip(Q,OFF): s=step_goal(e,s,np.array([tx+o[0],ty+o[1],o[2]]),q,1,35);s=step_goal(e,s,np.array([tx+o[0],ty+o[1],o[2]]),q,-1,2)
 if g(s,'robot','grasp_active')<.5: e.close();return ('nograsp',)
 q=np.array([g(s,'robot','joint_%d'%i) for i in range(1,8)]);c=xy(s,'cube0');u=np.array([.6-c[0],-c[1]]);u/=np.linalg.norm(u);v=np.array([-u[1],u[0]]);side=c-u*dist+v*lat
 # optional dogleg puts box laterally clear before approaching staging location
 if route:
  dog=side+v*route
  for _ in range(12):
   p=xy(s,'box0');b=np.array([g(s,'robot','pos_base_x')+dog[0]-p[0],g(s,'robot','pos_base_y')+dog[1]-p[1],g(s,'robot','pos_base_rot')]);s=step_goal(e,s,b,q,-1,1)
 for _ in range(15):
  p=xy(s,'box0');b=np.array([g(s,'robot','pos_base_x')+side[0]-p[0],g(s,'robot','pos_base_y')+side[1]-p[1],g(s,'robot','pos_base_rot')]);s=step_goal(e,s,b,q,-1,1)
 for _ in range(ls):
  p=xy(s,'box0');a=np.zeros(11,np.float32);a[:2]=np.clip(np.array([.6,0])-p,-.08,.08);a[4]=lift;a[10]=-1;s,r,t,tr,_=e.step(a)
 for _ in range(carry):
  c=xy(s,'cube0');a=np.zeros(11,np.float32);a[:2]=np.clip(np.array([.6,0])-c,-.1,.1);a[10]=-1;s,r,t,tr,_=e.step(a)
 for _ in range(12):
  c=xy(s,'cube0');a=np.zeros(11,np.float32);a[:2]=np.clip(np.array([.6,0])-c,-.08,.08);a[4]=.1;a[10]=-1;s,r,t,tr,_=e.step(a)
 for _ in range(8):
  c=xy(s,'cube0');a=np.zeros(11,np.float32);a[:2]=np.clip(np.array([.6,0])-c,-.08,.08);a[6]=-.1;a[10]=-1;s,r,t,tr,_=e.step(a)
 out=(round(g(s,'cube0','pose_x'),3),round(g(s,'cube0','pose_y'),3),round(g(s,'cube0','pose_z'),3),round(g(s,'box0','pose_z'),3))
 e.close();return out
if __name__=='__main__':
 seed=int(sys.argv[1])
 for dist in (.14,.18,.22,.26,.30):
  for lat in (-.12,-.06,0,.06,.12):
   out=run(seed,dist,lat,0,-.1)
   print(seed,dist,lat,0,-.1,out,flush=True)
