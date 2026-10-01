from env_client import make_env
from approach import GeneratedApproach

def v(s,n,f): return float(s.get(s.get_object_from_name(n),f))
for seed,count in [(4,1),(9,1),(4,0)]:
 e=make_env(); s,info=e.reset(seed=seed,options={'object_count':count})
 p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
 print('INIT',seed,count,[(n,round(v(s,n,'pose_x'),3),round(v(s,n,'pose_y'),3),round(v(s,n,'pose_z'),3)) for n in s.get_object_names() if n!='robot'])
 last=(-1,-1)
 for k in range(160):
  a=p.get_action(s);s,r,t,tr,i=e.step(a)
  key=(p.route_index,p.phase)
  if key!=last:
   print('EV',seed,count,k,key,'held',v(s,'robot','grasp_active'),'base',round(v(s,'robot','pos_base_x'),2),round(v(s,'robot','pos_base_y'),2),'q',[round(x,2) for x in p.q(s)],'box',[round(v(s,'box0','pose_'+c),2) for c in 'xyz'])
   last=key
  if t or tr: break
 e.close()
