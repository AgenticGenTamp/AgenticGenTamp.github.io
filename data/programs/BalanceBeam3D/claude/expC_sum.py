import re,glob,numpy as np,collections
D=collections.defaultdict(list)
for f in glob.glob('/sandbox/expC_logs/*.txt'):
    for ln in open(f):
        m=re.match(r"LAT=(\S+) PERP=(\S+) seed(\d+) bi(\d+) ok=(\d) knock=(\S+) off=\[(\S+),(\S+),(\S+)\]",ln)
        if m: D[(float(m.group(1)),float(m.group(2)))].append((int(m.group(3)),int(m.group(4)),int(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8))))
for k in sorted(D):
    a=np.array(D[k]); ok=a[:,2]; n=len(a)
    s=a[ok==1]
    print(f"LAT={k[0]:.3f} PERP={k[1]:.3f} n={n} succ={ok.mean()*100:.0f}% ({int(ok.sum())}/{n}) "
          f"offx={s[:,4].mean():+.4f}+-{s[:,4].std():.4f} offy={s[:,5].mean():+.4f}+-{s[:,5].std():.4f} "
          f"knock_all={a[:,3].mean():.4f} knock_fail={(a[ok==0][:,3].mean() if (ok==0).any() else float('nan')):.4f}")
    for bi in (0,54,70):
        b=a[a[:,1]==bi]
        print(f"   bi{bi}: {int(b[:,2].sum())}/{len(b)}")
