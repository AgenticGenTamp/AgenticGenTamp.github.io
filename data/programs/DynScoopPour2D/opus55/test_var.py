import sys
from multiprocessing import Pool
import approach
from test_approach import run
# usage: test_var.py a b KEY=VAL ...
def f(args):
    s, kv = args
    for k,v in kv.items(): setattr(approach.GeneratedApproach, k, v)
    try: return s, run(s)
    except Exception as e: return s, ('ERR', str(e)[:80])
if __name__=='__main__':
    a=int(sys.argv[1]); b=int(sys.argv[2]); kv={}
    for x in sys.argv[3:]:
        k,v=x.split('='); 
        try: v=float(v)
        except: pass
        kv[k]=v
    with Pool(10) as p: res=p.map(f, [(s,kv) for s in range(a,b)])
    ok=0; steps=[]
    for s,r in res:
        print(s, r)
        if r[0] is True: ok+=1; steps.append(r[1])
    print(kv, 'success', ok, '/', len(res), 'mean steps', sum(steps)/max(1,len(steps)))
