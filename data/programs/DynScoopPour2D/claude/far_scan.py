from env_client import make_env
from scoop_lib import Scoop
env = make_env()
res=[]
for s in range(61):
    c = Scoop(env, s)
    hx = c.g('x', c.H); hy=c.g('y',c.H); th=c.g('theta',c.H)
    res.append((s, round(hx,4), round(hy,4), round(th,4)))
far=[r for r in res if r[1]>3.40]
print("FAR:", far)
print("maxall", max(r[1] for r in res), "n>3.30:", sum(1 for r in res if r[1]>3.30))
print("all:", res)
env.close()
