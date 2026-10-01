import sys; sys.path.insert(0,"scratch"); from h import *
for s in range(6):
    o,info=env.reset(seed=s)
    print(s,info, rstate(o).round(3))
    for b in blocks(o): print(" ",b.name,bpose(o,b).round(3),round(byaw(o,b),3))
    for sf in o.get_objects(T("surface")): print(" ",sf.name,[round(o.get(sf,x),3) for x in ["pose_x","pose_y","pose_z","half_extent_x","half_extent_y","half_extent_z"]])
