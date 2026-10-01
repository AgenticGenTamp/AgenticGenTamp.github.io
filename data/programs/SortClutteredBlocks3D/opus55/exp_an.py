import json,glob,collections,numpy as np,sys
R=collections.defaultdict(list)
for f in glob.glob("exp_res_*.json"):
    d=json.load(open(f)); R[d["tag"]].append(d)
print("tag       n  placed(mean) succ fail  rate  [lift>sel putback desc>sel]")
allrec=[]
for tag,ds in sorted(R.items()):
    t=collections.Counter()
    for d in ds: t.update(d["trans"]); allrec+= [dict(r,tag=tag) for r in d["recs"]]
    s=t["lift>transport"]; f=t["lift>select"]+t["lift>putback"]+t["descend>select"]
    print(f"{tag:8s} {len(ds):2d} {np.mean([d['placed'] for d in ds]):5.1f} {sorted(d['placed'] for d in ds)} {s:4d} {f:4d} {s/max(1,s+f):5.2f}  [{t['lift>select']} {t['lift>putback']} {t['descend>select']}]")
if len(sys.argv)>1:
    recs=[r for r in allrec if r["tag"] in sys.argv[1].split(",")]
    bins=[-1,-0.01,-0.005,-0.002,0,0.003,0.006,0.01,1]
    print("clr bin        n  succ  lift>sel putback nbr_mv>3mm")
    for lo,hi in zip(bins[:-1],bins[1:]):
        rr=[r for r in recs if lo<=r["clr"]<hi]
        if not rr: continue
        print(f"[{lo:+.3f},{hi:+.3f}) {len(rr):3d} {sum(r['out']=='lift>transport' for r in rr)/len(rr):5.2f} {sum(r['out']=='lift>select' for r in rr):4d} {sum(r['out']=='lift>putback' for r in rr):4d} {sum(r.get('nbr_mv',0)>0.003 for r in rr):4d}")
    for o in ("lift>transport","lift>select","lift>putback"):
        rr=[r for r in recs if r["out"]==o]
        if rr: print(o,len(rr),"nbr_mv>3mm frac %.2f"%np.mean([r.get('nbr_mv',0)>0.003 for r in rr]),"median nbr_mv %.4f"%np.median([r.get('nbr_mv',0) for r in rr]))
