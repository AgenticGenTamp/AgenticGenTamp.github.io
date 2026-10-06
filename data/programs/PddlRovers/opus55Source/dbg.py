import sys, math
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from approach import GeneratedApproach
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 40
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=seed)
a = GeneratedApproach(env.action_space, env.observation_space, {})
a.reset(state, info)
print("assign", a.assign)
for r in range(2): print(r, [(round(p[0],2),round(p[1],2),op) for p,op in a.plans[r]])
rv,_,objs,sams,obst=a._parse(state)
for o in objs: print("obj", round(o['x'],2), round(o['y'],2))
for s in sams: print("sam", round(s['x'],2), round(s['y'],2), s['is_soil'])
for k in range(N):
    act=a.get_action(state)
    state,_,term,_,_=env.step(act)
    rv,_,objs,sams,_=a._parse(state)
    print(k, [round(float(x),2) for x in act], [(round(v['x'],2),round(v['y'],2),round(v['theta'],2),v['store_full'],v['calibrated'],v['at_home']) for v in rv],
      [(o['have_image_rover0'],o['have_image_rover1'],o['received_image']) for o in objs])
    if term: print("DONE"); break
