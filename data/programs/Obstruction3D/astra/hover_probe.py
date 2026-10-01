exec(open('probe_ik.py').read().split('for L in [.12')[0])
s,_=E.reset(seed=0);bp=np.array([s.get(b,'pose_'+v) for v in 'xyz'])
for yaw in [0,1.57079632679]:
 for off in [0,-.03,.03,-.06,.06]:
  s,_=E.reset(seed=0)
  goal=np.r_[bp[0]-.5+off,bp[1]-.001,0,solve(.5,.35,.2)];goal[-1]=yaw
  move(goal)
  for z in np.arange(.35,.05,-.002):
   goal[3:]=solve(.5,z,.2);goal[-1]=yaw
   ok=move(goal)
   a=np.zeros(11);a[-1]=1;s,*_=E.step(a)
   a[-1]=-1;s,*_=E.step(a)
   if s.get(r,'grasp_active'):
    print('GRASP',yaw,off,z,rv(s),flush=True)
    print({f:s.get(r,f) for f in E.observation_space.type_features[r.type]},flush=True);E.close();raise SystemExit
   if not ok:print('BLOCKED',yaw,off,z,flush=True);break
E.close()
