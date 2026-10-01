import sys, time
from env_client import make_env
from approach import GeneratedApproach
env = make_env()
obs, info = env.reset(seed=int(sys.argv[1]), options={'object_count': int(sys.argv[2])})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t0=time.time(); ap.reset(obs, info); tr=time.time()-t0
ta=0; te=0; n=0; mx=0
while n < 1000:
    t0=time.time(); a = ap.get_action(obs); d=time.time()-t0; ta+=d; mx=max(mx,d)
    t0=time.time(); obs, r, term, trunc, info = env.step(a); te+=time.time()-t0; n += 1
    if term: break
print('steps',n,'reset %.2f'%tr,'agent %.2f'%ta,'max %.3f'%mx,'env %.2f'%te)
