import sys
from test_approach import *
env = make_env()
n, s = int(sys.argv[1]), int(sys.argv[2])
ok, steps, dt, st = run(env, s, n, max_steps=int(sys.argv[3]) if len(sys.argv)>3 else None, verbose=True)
print(ok, steps)
