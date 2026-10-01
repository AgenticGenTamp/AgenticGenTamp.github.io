import numpy as np
from env_client import make_env
QS=[np.array(x) for x in [[.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593],[4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916],[4.170779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434]]]
O=[[-.799987478,-.347800459,2.104792961],[.126212298,-.083534270,-3.139703362],[-.428185141,.054943428,2.978131848]]
def v(s,n,f):return float(s.get(s.get_object_from_name(n),f))
def go(e,s,q,yaw,base=None,n=35,close=False):
 for _ in range(n):
  a=np.zeros(11,np.float32);a[2]=np.clip(yaw-v(s,'robot','pos_base_rot'),-.2,.2);a[3:10]=np.clip(q-np.array([v(s,'robot','joint_'+str(i)) for i in range(1,8)]),-.2,.2)
  if base is not None:a[:2]=np.clip(np.array(base)-[v(s,'robot','pos_base_x'),v(s,'robot','pos_base_y')],-.2,.2)
  a[10]=-1 if close else 1;s=e.step(a)[0]
 return s
for seed,count in [(4,0),(4,1),(9,1)]:
 e=make_env();s,_=e.reset(seed=seed,options={'object_count':count});tx=v(s,'box0','pose_x');ty=v(s,'box0','pose_y')
 # pose arm at wp0 without entering table, then approach only legal outer waypoint 1
 s=go(e,s,QS[0],O[0][2]);s=go(e,s,QS[1],O[1][2]);s=go(e,s,QS[1],O[1][2],[tx+O[1][0],ty+O[1][1]])
 s=go(e,s,QS[2],O[2][2],[tx+O[2][0],ty+O[2][1]]);s=go(e,s,QS[2],O[2][2],[tx+O[2][0],ty+O[2][1]],5,True)
 print(seed,count,'held',v(s,'robot','grasp_active'),'base',round(v(s,'robot','pos_base_x'),2),round(v(s,'robot','pos_base_y'),2))
 e.close()
