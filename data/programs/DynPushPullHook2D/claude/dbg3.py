import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); N=int(sys.argv[2])
env = make_env(); ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs,info)
for i in range(N):
    obs,r,term,tr,info = env.step(ap.get_action(obs))
    if term: print("SOLVED",i); break
tf = {t.name: list(f) for t,f in env.observation_space.type_features.items()}
for name in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(name)
    d={f: round(float(obs.get(o,f)),3) for f in tf[o.type.name] if f in ('x','y','theta','width','height','held','arm_joint')}
    print(name, d)
env.close()
