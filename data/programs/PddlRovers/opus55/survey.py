from env_client import make_env
import collections
env = make_env()
T=lambda t: env.observation_space.get_type(t)
cnt=collections.Counter(); objxy=set(); landers=set(); mounds=set()
for seed in range(60):
    obs,info=env.reset(seed=seed)
    cnt[info['object_count']]+=1
    for o in obs.get_objects(T('objective')): objxy.add((round(obs.get(o,'x'),1),round(obs.get(o,'y'),1)))
    l=obs.get_objects(T('lander'))[0]; landers.add((round(obs.get(l,'x'),2),round(obs.get(l,'y'),2)))
    for o in obs.get_objects(T('obstacle')):
        if obs.get(o,'half_x')>0.1: mounds.add((o.name,obs.get(o,'x'),obs.get(o,'y')))
    r=[(round(obs.get(o,'x'),2),round(obs.get(o,'y'),2)) for o in obs.get_objects(T('rover'))]
    if seed<3: print(r)
print(cnt); print(sorted(objxy)); print(landers); print(sorted(mounds))
