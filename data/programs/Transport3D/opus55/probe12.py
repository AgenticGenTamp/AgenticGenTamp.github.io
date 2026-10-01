from env_client import make_env
import collections
env=make_env(); cnt=collections.Counter()
for s in range(200):
    obs,info=env.reset(seed=s); cnt[info['object_count']]+=1
print(cnt)
env.close()
