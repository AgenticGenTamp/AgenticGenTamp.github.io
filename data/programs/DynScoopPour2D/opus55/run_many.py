import sys
from multiprocessing import Pool
from test_approach import run
def f(s):
    try: return s, run(s)
    except Exception as e: return s, ('ERR', str(e)[:80])
if __name__=='__main__':
    a=int(sys.argv[1]); b=int(sys.argv[2])
    with Pool(10) as p:
        res=p.map(f, range(a,b))
    ok=0; steps=[]
    for s,r in res:
        print(s, r)
        if r[0] is True: ok+=1; steps.append(r[1])
    print('success', ok, '/', len(res), 'mean steps', sum(steps)/max(1,len(steps)))
