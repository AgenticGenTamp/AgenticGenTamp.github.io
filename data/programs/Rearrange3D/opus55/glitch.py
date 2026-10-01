import approach, numpy as np
from multiprocessing import Pool
from test_approach import run
def f(s):
    try:
        r=run(s,False); return (s,r['term'],r['steps'],round(r['time'],1))
    except Exception as e: return (s,'ERR',repr(e)[:100])
if __name__=='__main__':
    with Pool(13) as p:
        for x in p.map(f,[27,32,36,46,88,138,49,53,60,101,125,146,51]): print(x)
