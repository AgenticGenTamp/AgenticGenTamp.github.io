from env_client import make_env
import numpy as np, json
env = make_env()
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
log=[]
def snap(obs, rew):
    d={'rew':float(rew)}
    for i in range(1,5):
        o=obs.get_object_from_name(f'cube{i}')
        d[f'c{i}']=[float(obs.get(o,f)) for f in 'xyz']
    return d
for t in range(200):
    a = rng.uniform(-0.1,0.1,11).astype(np.float32); a[10]=0.0
    obs, rew, term, trunc, info = env.step(a)
    log.append(snap(obs,rew))
json.dump(log, open('rollout0.json','w'))
print('final', log[-1])
print('rews', sorted(set(round(l['rew'],4) for l in log))[:10])
env.close()
