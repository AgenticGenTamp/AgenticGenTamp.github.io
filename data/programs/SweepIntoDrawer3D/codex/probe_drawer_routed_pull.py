import numpy as np
from env_client import make_env


env = make_env()
o, _ = env.reset(seed=0)

def step(a):
    global o
    o, r, term, trunc, _ = env.step(np.asarray(a, np.float32))
    return r

# Clear the east corner first, then translate west along the negative-y face.
for _ in range(9):
    a = np.zeros(11); a[1] = -.1; a[2] = -.1
    step(a)
for _ in range(7):
    a = np.zeros(11); a[0] = -.1; a[2] = -.1
    step(a)
print("aligned", o[125:128].round(3), "q",o[128:135].round(3),"draw",o[103:109].round(4))

# Let the base advance until it makes contact, close, then pull back.
for phase, n, vy, grip in [("advance", 4, .08, 0), ("close", 2, 0, 1), ("pull", 5, -.08, 1)]:
    for k in range(n):
        a = np.zeros(11); a[1] = vy; a[10] = grip
        r = step(a)
        print(phase,k+1,"base",o[125:128].round(3),"g",round(float(o[135]),2),"draw",o[103:109].round(4),"r",r)
print("FINAL_STATE", o.tolist())
env.close()
