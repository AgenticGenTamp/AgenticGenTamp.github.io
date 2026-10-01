import io, contextlib, re, approach, numpy as np
from multiprocessing import Pool
from test_approach import run
def f(s):
    approach.DEBUG=True
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): run(s,False)
    out=[]
    for m in re.finditer(r"seg dq \[([^\]]*)\] n (\d+)", buf.getvalue()):
        dq=np.array(m.group(1).split(),float); n=int(m.group(2))
        if n>=12: out.append((s,n,dq.round(2).tolist()))
    return out
if __name__=='__main__':
    with Pool(12) as p:
        for o in p.map(f,[s for s in range(0,26) if s not in (27,)]):
            for s,n,dq in o:
                j=int(np.argmax(dq)); print(s,n,'argmax j%d'%(j+1),dq)
