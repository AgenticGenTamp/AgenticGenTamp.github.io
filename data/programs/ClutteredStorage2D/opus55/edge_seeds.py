from env_client import make_env
env = make_env()
out=[]
for s in range(0, 600):
    obs, info = env.reset(seed=s)
    sh = obs.get_object_from_name('shelf')
    x1 = obs.get(sh,'x1'); w1 = obs.get(sh,'width1')
    if x1 < 0.2 or x1 + w1 > 4.8: out.append((s, round(x1,3), round(x1+w1,3), info['object_count']))
print(out)
