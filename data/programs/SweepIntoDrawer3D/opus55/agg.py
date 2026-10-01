import collections, sys
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    if not l.startswith('('): continue
    t=eval(l); d[str(t[1])].append((t[0],t[2],t[3]))
for k,v in d.items():
    v.sort(); print(k, "succ", sum(x[1] for x in v), "/", len(v), "mean", round(sum(x[2] for x in v)/len(v),1), "max", max(x[2] for x in v), "fails", [x[0] for x in v if not x[1]])
