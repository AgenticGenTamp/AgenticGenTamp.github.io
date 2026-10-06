import sys, numpy as np
from simutil import make_env, obj, robot_q
from approach import GeneratedApproach
N=int(sys.argv[1]); sd=int(sys.argv[2]); env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=sd,options={'object_count':N}); ap.reset(s,info)
for k in range(int(sys.argv[3])):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    if k%int(sys.argv[4] if len(sys.argv)>4 else 10)==0 or term:
        b,q=robot_q(s); p,_=ap.kin.fk(b,q)
        cz=np.mean([obj(s,'cube_%d'%i)[2] for i in range(N)]); cy=np.mean([obj(s,'cube_%d'%i)[1] for i in range(N)])
        print(k, ap.phase, np.round(ap.last_err,3),'ee',np.round(p,3),'Y',np.round(obj(s,'bin_yellow_0'),2),'G',np.round(obj(s,'bin_green_0')[:3],3),'cubes y,z',round(cy,3),round(cz,3), 'grip', s.get(s.get_object_from_name('robot'),'pos_gripper'))
    if term: break
for i in range(N): print(np.round(obj(s,'cube_%d'%i)[:3],3))
