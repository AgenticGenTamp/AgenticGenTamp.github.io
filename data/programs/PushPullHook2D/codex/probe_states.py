from env_client import make_env

env = make_env()
print('max', env.max_steps)
for seed in range(20):
    s, info = env.reset(seed=seed)
    print(seed, 'robot', s[[0,1,2,4,5,6]], 'hook', s[[9,10,11,17,18,19]], 'button', s[[20,21,28]], 'target', s[[29,30,37]])
env.close()
