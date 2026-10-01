import json,numpy as np
d=json.load(open("map.json")); free=set(map(tuple,d["free"]))
xs=[c[0] for c in free]; ys=[c[1] for c in free]
print("x range",min(xs),max(xs),"y",min(ys),max(ys))
for j in range(max(ys),min(ys)-1,-1):
    row="".join("." if (i,j) in free else "#" for i in range(min(xs),max(xs)+1))
    print(f"{j*0.1:5.1f} "+row)
print("      "+"".join(str(abs(int(round(i*0.1))%10)) for i in range(min(xs),max(xs)+1)))
