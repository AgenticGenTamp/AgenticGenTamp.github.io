from env_client import make_env
E=make_env()
for seed in range(12):
 s,i=E.reset(seed=seed)
 R=s.get_objects(E.observation_space.get_type('kin_robot'))[0];H=s.get_objects(E.observation_space.get_type('hook'))[0]
 print(seed,i,'robot',[round(s.get(R,k),3) for k in ('x','y','theta','arm_joint')],'hook',[round(s.get(H,k),3) for k in ('x','y','theta','length_side1','length_side2')])
E.close()
