import re,glob,ast
for f in sorted(glob.glob('diag_head/o_*.txt')):
    L=open(f).read().splitlines(); cur=None
    for l in L:
        if ' CLOSE ' in l:
            nb=ast.literal_eval(re.search(r"nb (\[.*?\]) base",l).group(1))
            m=re.search(r'CLOSE (\w+) along(\S+) perp(\S+) dz(\S+) phi(\S+) clr(\S+)',l)
            cur=(l.split()[0],)+m.groups()+(nb,)
        elif 'lift ->' in l and cur:
            ok='transport' in l
            nb=[(n,a,p,z) for n,a,p,z in cur[-1] if abs(a)<0.045 and abs(p)<0.04]
            print(f[-6:-4],'OK ' if ok else 'FAIL',cur[0],cur[1],'dz',cur[4],'clr',cur[6],'nb(al,pe,dz)',nb); cur=None
