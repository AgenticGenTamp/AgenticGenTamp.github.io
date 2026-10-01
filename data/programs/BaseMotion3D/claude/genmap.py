import json
d=json.load(open("map.json")); free=set(map(tuple,d["free"]))
cols={}
for (i,j) in free: cols.setdefault(i,[]).append(j)
parts=[]
for i in sorted(cols):
    ys=sorted(cols[i]); runs=[]; s=ys[0]; p=ys[0]
    for y in ys[1:]:
        if y==p+1: p=y
        else: runs.append((s,p)); s=y; p=y
    runs.append((s,p))
    parts.append(f"{i}:"+",".join(f"{a}~{b}" for a,b in runs))
enc=";".join(parts)
print(len(enc))
open("map_enc.txt","w").write(enc)
