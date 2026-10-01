exec(open('side_probe.py').read().split('for ang in [-np.pi')[0])
for name in ['obstruction0','obstruction1']:
 for dx,dy in [(0,0),(-.02,0),(.02,0),(0,-.02),(0,.02)]:
  s,_=E.reset(seed=0);ob=s.get_object_from_name(name);p=np.array([s.get(ob,'pose_'+v) for v in 'xyz'])
  goal=np.r_[p[0]-.5+dx,p[1]-.001+dy,0,solve(.5,.4,np.pi/4,.061525)];move(goal)
  for z in np.arange(.35,.10,-.005):
   goal[3:]=solve(.5,z,np.pi/4,.061525);ok=move(goal)
   a=np.zeros(11);a[-1]=-1;s,*_=E.step(a)
   if s.get(r,'grasp_active'):
    print('GRASP',name,dx,dy,z,rv(),flush=True)
    print({o.name:{f:s.get(o,f) for f in E.observation_space.type_features[o.type]} for o in [r,ob]},flush=True);break
   if not ok:print('BLOCKED',name,dx,dy,z,flush=True);break
  if s.get(r,'grasp_active'):break
E.close()
