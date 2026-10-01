from env_client import make_env
import collections
env = make_env()
cnt = collections.Counter()
names = collections.Counter()
for s in range(30):
    obs, info = env.reset(seed=s)
    ns = tuple(sorted(obs.get_object_names()))
    cnt[info.get('object_count')] += 1
    names[ns] += 1
print(cnt)
for k,v in names.items(): print(v, k)
env.close()
