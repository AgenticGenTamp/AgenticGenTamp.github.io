import sys, subprocess, collections
from multiprocessing import Pool
def one(args):
    c,s=args
    out=subprocess.run(["/robocode/.venv/bin/python","phases.py",str(c),str(s),str(s+1)],capture_output=True,text=True).stdout
    return [l for l in out.splitlines() if l and l[0].isdigit()][0]
c=int(sys.argv[1]); seeds=range(int(sys.argv[2]),int(sys.argv[3]))
tot=collections.Counter(); n=0
with Pool(16) as p:
    for l in p.map(one,[(c,s) for s in seeds]):
        n+=1; print(l.split()[0], l.split()[1], end=" | ")
        for tok in l.split()[2:]:
            k,v=tok.rsplit(":",1); tot[k.split("/")[0] if not k.startswith(("approach","moves")) else "/".join(k.split("/")[:2])]+=int(v)
for k,v in sorted(tot.items(), key=lambda x:-x[1]): print(f"{k:20s} {v/n:6.1f}")
