from env_client import make_env
E=make_env();s,i=E.reset(seed=1)
o=s.get_object_from_name('robot')
print([(f,float(s.get(o,f))) for f in ['pos_arm_joint'+str(k) for k in range(1,8)]+['pos_gripper']],flush=True)
E.close()
