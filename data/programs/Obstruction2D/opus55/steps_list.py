import sys
from test_approach import run
s0,s1=int(sys.argv[1]),int(sys.argv[2])
for s in range(s0,s1):
    ok,n,c=run(s)
    print(s,ok,n,c,flush=True)
