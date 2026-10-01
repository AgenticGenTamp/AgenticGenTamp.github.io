import sys, io, contextlib, re, approach
from test_approach import run
for kv in sys.argv[2:]:
    k,v=kv.split('='); setattr(approach,k,float(v))
approach.DEBUG=True
buf=io.StringIO()
with contextlib.redirect_stdout(buf): r=run(int(sys.argv[1]),False)
print(' | '.join('%s:%s/%s'%m.groups() for m in re.finditer(r"done (\w+) k (\d+) n (\d+)", buf.getvalue())), r['steps'], r['term'])
