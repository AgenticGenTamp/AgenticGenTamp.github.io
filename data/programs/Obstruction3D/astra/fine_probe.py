exec(open('probe_ik.py').read().split('for L in [.12')[0])
for L in [.185,.19,.195,.20,.21,.22]:
 s,_=E.reset(seed=0)
 bp=np.array([s.get(b,'pose_'+v) for v in 'xyz'])
 goal=np.r_[bp[0]-.5,bp[1]-.001,0,solve(.5,.35,L)]
 move(goal)
 for z in np.arange(.18,.04,-.002):
  goal[3:]=solve(.5,z,L)
  ok=move(goal,-1)
  if s.get(r,'grasp_active'):
   print('GRASP',L,z,rv(s),flush=True)
   print({f:s.get(r,f) for f in E.observation_space.type_features[r.type]},flush=True)
   E.close();raise SystemExit
  if not ok:
   print('BLOCKED',L,z,'tip_model',fk(rv(s)[3:])[:3,3]+fk(rv(s)[3:])[:3,:3]@[0,0,-L],flush=True)
   break
E.close()
