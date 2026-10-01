import sys, numpy as np, approach
from env_client import make_env
env=make_env(); ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=int(sys.argv[1]), **({'options':{'object_count':int(sys.argv[3])}} if len(sys.argv)>3 else {})); ap.reset(obs,info)
def summ(obs):
    out=[]
    for n in sorted(obs.get_object_names()):
        o=obs.get_object_from_name(n)
        if n=='robot': out.append('R(%.2f,%.2f,%.2f g%d)'%tuple(obs.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot','grasp_active']))
        elif n!='table': out.append('%s(%.3f,%.3f,%.3f)'%(n,obs.get(o,'pose_x'),obs.get(o,'pose_y'),obs.get(o,'pose_z')))
    return ' '.join(out)
for t in range(int(sys.argv[2]) if len(sys.argv)>2 else 300):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    print(t, np.round(a,2).tolist(), summ(obs))
    if term: print('TERM'); break
