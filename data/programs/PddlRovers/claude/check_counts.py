from env_client import make_env
env = make_env()
cnt={}
for s in range(40):
    _,info = env.reset(seed=s)
    c=info['object_count']; cnt[c]=cnt.get(c,0)+1
print(cnt)
env.close()
