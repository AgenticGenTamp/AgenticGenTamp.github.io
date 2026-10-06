from shapely.geometry import box
import approach
from layout import place_poly, clearance, RACK_INNER
from layouts_cache import LAYOUTS
S=("cuboid",(0.05,0.05,0.01)); E=("tri",("equilateral",0.1,0.1)); R=("tri",("right",0.1,0.1))
D={"S":S,"E":E,"R":R}; inner=box(*RACK_INNER)
for key,lay in LAYOUTS.items():
    specs=[D[c] for c in key]
    res=[]
    for fx,fy in ((0,0),(0,1),(1,0),(1,1)):
        v=approach._reflect_layout([tuple(x) for x in lay],specs,0.3,0.0,bool(fx),bool(fy))
        if v is None: res.append(None); continue
        polys=[place_poly(approach._footprint_of(sp),*o) for sp,o in zip(specs,v)]
        res.append(round(clearance(polys,inner),4))
    print(key,res)
