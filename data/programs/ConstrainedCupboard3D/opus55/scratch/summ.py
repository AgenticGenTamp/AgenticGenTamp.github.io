import sys,re,glob
for f in sorted(glob.glob(sys.argv[1])):
    s=''
    for l in open(f):
        m=re.match(r'Y (\S+) Z (\S+) tipx (\S+) raw (.*) steps',l)
        if not m: continue
        t=m.group(3)
        if t=='None': c='?' if 'fail' in m.group(4) or 'noik' in m.group(4) else '.'
        else:
            x=float(t); c='#' if x<1.86 else ('+' if x<1.885 else ('o' if x<1.92 else '.'))
        s+=c
    print('%-18s %s'%(f,s), [l.strip() for l in open(f) if 'LOG' in l])
