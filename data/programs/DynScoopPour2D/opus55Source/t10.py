from common import make_env, hook
env=make_env()
out=[]
for sd in range(60):
    s,_=env.reset(seed=sd,options={'object_count':3}); H=hook(s)
    if H['x']>3.39: out.append((sd,round(H['x'],3),round(H['y'],3)))
print(out)
