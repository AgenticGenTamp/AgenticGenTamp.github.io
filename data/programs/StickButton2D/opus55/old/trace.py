import sys, os
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
seed=int(sys.argv[1]); cnt=os.environ.get("COUNT")
obs,info=env.reset(seed=seed, options=({"object_count":int(cnt)} if cnt else None))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
ap._read(obs)
print("stick", [round(v,3) for v in ap.stick], "robot", round(ap.rx,3), round(ap.ry,3), round(ap.rth,3))
print("buttons", [(b[0], round(b[1],3), round(b[2],3)) for b in ap.buttons])
last=None
for n in range(1000):
    a=ap.get_action(obs)
    info=(str(ap.plan), ap.grasped, len(ap.buttons), getattr(ap,'cur_target',None) and ap.cur_target[1])
    if info!=last:
        print(n, "pos", round(ap.rx,3), round(ap.ry,3), round(ap.rth,3), "plan", info)
        last=info
    obs,r,term,trunc,_=env.step(a)
    if term: print("done", n+1); break
