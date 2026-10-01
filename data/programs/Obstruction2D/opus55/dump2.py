from env_client import make_env
env = make_env()
import collections
cnt=collections.Counter()
for seed in range(0,60):
    obs, info = env.reset(seed=seed)
    cnt[info.get('object_count')]+=1
    r=[o for o in obs.get_objects(env.observation_space.get_type('crv_robot'))][0]
    ths=[round(float(obs.get(o,'theta')),3) for o in obs.get_objects(env.observation_space.get_type('rectangle'))]
    ys=[round(float(obs.get(o,'y')),3) for o in obs.get_objects(env.observation_space.get_type('rectangle'))]
    if any(ths) or set(ys)-{0.1,0.0} or abs(obs.get(r,'theta')+1.5708)>1e-3: print(seed, ths, ys, obs.get(r,'theta'))
print(cnt)
