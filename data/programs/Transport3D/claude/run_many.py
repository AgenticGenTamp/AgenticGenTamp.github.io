import sys, concurrent.futures as cf
from run_eval import run
cases=[]
for s in range(int(sys.argv[1]), int(sys.argv[2])):
    for oc in ([int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else [None]):
        cases.append((s,oc))
def f(c):
    try: return run(c[0], c[1])
    except Exception as e: return {'seed':c[0],'oc':c[1],'err':str(e)[:80]}
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    res=list(ex.map(f,cases))
ok=sum(1 for r in res if r.get('term'))
print("SOLVED",ok,"/",len(res))
for r in res:
    if not r.get('term'): print("FAIL",r)
import numpy as np
st=[r['steps'] for r in res if r.get('term')]
print("steps mean",round(np.mean(st),1),"max",max(st) if st else None, "wall max", max(r.get('wall',0) for r in res))
