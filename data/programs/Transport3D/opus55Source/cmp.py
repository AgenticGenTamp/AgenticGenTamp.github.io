import sys,re,collections
def load(f):
    d=collections.defaultdict(list); fails=[]
    for l in open(f):
        m=re.search(r"k=(\d+) seed=(\d+) ok=(\w+) steps=(\d+)",l)
        if m:
            k,s,ok,st=int(m[1]),int(m[2]),m[3],int(m[4]); d[k].append(st)
            if ok!='True': fails.append((k,s))
    return d,fails
for f in sys.argv[1:]:
    d,fails=load(f)
    print(f, {k:(len(v),round(sum(v)/len(v),1)) for k,v in sorted(d.items())}, 'fails',fails)
