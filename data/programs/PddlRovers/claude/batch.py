import sys, numpy as np
from env_client import make_env
import approach
seeds=range(int(sys.argv[1]), int(sys.argv[2]))
cap=int(sys.argv[3]) if len(sys.argv)>3 else 200
env=make_env()
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
fails=[]; steps_l=[]
for seed in seeds:
    obs,info=env.reset(seed=seed)
    try:
        ap.reset(obs,info)
    except Exception as e:
        print(seed,"PLAN ERR",repr(e)[:100], flush=True); fails.append(seed); continue
    term=False; s=0
    while not term and s<cap:
        try:
            a=ap.get_action(obs)
        except Exception as e:
            print(seed,"ACT ERR",repr(e)[:100], flush=True); break
        obs,r,term,trunc,info=env.step(a); s+=1
    if term:
        steps_l.append(s); print(seed,info.get('object_count'),s, flush=True)
    else:
        fails.append(seed)
        miss=[]
        for n in sorted(obs.get_object_names()):
            o=obs.get_object_from_name(n)
            if n.startswith('objective') and float(obs.get(o,'received_image'))<0.5: miss.append(n)
            if n.startswith('rover'):
                miss.append((n,round(float(obs.get(o,'x')),2),round(float(obs.get(o,'y')),2),
                             float(obs.get(o,'store_full')),float(obs.get(o,'at_home'))))
        need=[t for k in range(2) for t in ap.tasks[k]]
        print(seed,info.get('object_count'),"FAIL",miss,[ (t['kind'],t['obj']) for t in need], flush=True)
print("solved",len(steps_l),"/",len(list(seeds)),"mean",round(np.mean(steps_l),1) if steps_l else None,"fails",fails)
env.close()
