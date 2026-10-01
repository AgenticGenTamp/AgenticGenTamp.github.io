import sys, io, contextlib, re, collections, approach
from test_approach import run
approach.DEBUG=True
tot=collections.Counter(); ext=collections.Counter()
for s in [0,1,2,3,4,6,7,8]:
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): run(s,False)
    for m in re.finditer(r"done (\w+) k (\d+) n (\d+)", buf.getvalue()):
        kind,k,n=m.group(1),int(m.group(2)),int(m.group(3))
        tot[kind]+=k; ext[kind]+=max(0,k-n)
print('total steps per kind',dict(tot)); print('extra(beyond traj)',dict(ext))
