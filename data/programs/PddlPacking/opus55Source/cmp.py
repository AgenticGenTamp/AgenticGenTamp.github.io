import sys, re
def load(tag,k):
    d={}
    try:
        for l in open(f'logs/{tag}/k{k}.txt'):
            m=re.match(r'(\d+) \((True|False), (\d+), ([\d.]+)',l)
            if m: d[int(m[1])]=(m[2]=='True',int(m[3]),float(m[4]))
    except FileNotFoundError: pass
    return d
tags=sys.argv[1:]
for k in range(1,10):
    ds=[load(t,k) for t in tags]
    if not ds[0]: continue
    print(f'k{k}', ' | '.join(' '.join(f"{d.get(s,(0,0,0))[1]:3d}" for s in range(20)) for d in ds))
