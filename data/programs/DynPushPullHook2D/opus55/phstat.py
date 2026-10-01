import collections,sys
tot=collections.Counter(); cnt=0
for line in open(sys.argv[1]):
    p=line.split()
    if len(p)<4 or p[0]=='success': continue
    n=int(p[2]); ph=[(x.split('@')[0],int(x.split('@')[1])) for x in p[3:]]
    cnt+=1
    for i,(name,t) in enumerate(ph):
        e=ph[i+1][1] if i+1<len(ph) else n
        tot[name]+=e-t
print(' '.join(f'{k}:{round(v/cnt,1)}' for k,v in tot.most_common()))
