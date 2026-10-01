import re
rows=[]
for l in open('rew_7.out'):
    if not l.startswith('D '): continue
    m=re.match(r"D (\d+) (\S+) (\d) p \[\s*(\S+)\s+(\S+)\s+(\S+)\s*\] v \[\s*(\S+)\s+(\S+)\s+(\S+)\s*\] vmax (\S+) g (\S+)",l)
    rows.append((int(m[1]),m[2],int(m[3]),float(m[6]),float(m[9]),float(m[10]),float(m[11]),float(m[4]),float(m[5])))
# segments
seg=[];cur=None
for r in rows:
    if cur is None or r[2]!=cur[0] or r[1]!=cur[1]:
        cur=[r[2],r[1],r[0],r[0],[r[3]],[r[5]],[r[4]]]; seg.append(cur)
    else: cur[3]=r[0]; cur[4].append(r[3]); cur[5].append(r[5]); cur[6].append(r[4])
for s in seg:
    print(f"term={s[0]} {s[1]:12s} steps {s[2]}-{s[3]} z[{min(s[4]):.4f},{max(s[4]):.4f}] vz[{min(s[6]):.3f},{max(s[6]):.3f}] vmax[{min(s[5]):.3f},{max(s[5]):.3f}]")
