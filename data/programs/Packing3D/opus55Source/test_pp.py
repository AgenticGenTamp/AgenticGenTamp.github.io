import math, time, sys
import numpy as np
from planner import *
from layout import *
n=int(sys.argv[1]); seed=int(sys.argv[2])
tp=TwinPlanner(n)
s,_=tp.env.reset(seed=seed)
tp.state=s
inner=(0.21,-0.14,0.39,0.14)
parts=sorted([o for o in s if o.name.startswith("part")], key=lambda o:o.name)
fps=[]; 
for o in parts:
    if o.type.name=="Kinematic3DCuboid": fps.append(("cuboid",(s.get(o,"half_extent_x"),s.get(o,"half_extent_y"))))
    else: fps.append(("tri",(["equilateral","right"][int(s.get(o,"triangle_type"))],s.get(o,"side_a"),s.get(o,"side_b"))))
t=time.time(); lay,c=compute_layout(fps); print("layout clr",c, time.time()-t)
for o,(x,y,yaw) in zip(parts,lay):
    cen=s.get_object_pose(o.name).position
    print(o.name,"centroid",np.round(cen,3), "origin", np.round(s.get_object_origin_pose(o.name).position,3))
    tp.move_cartesian(Pose((cen[0],cen[1],0.30), tp.down_yaw(0)))
    tp.move_cartesian(Pose((cen[0],cen[1],cen[2]+0.07), tp.down_yaw(0)))
    tp.gripper(True)
    tf=tp.env._grasped_object_transform
    tp.move_cartesian(Pose((cen[0],cen[1],0.30), tp.down_yaw(0)))
    tgt=Pose((x,y,0.095+0.003), yaw_quat(yaw))
    eet=multiply_poses(tgt, tf.invert())
    tp.move_cartesian(Pose((eet.position[0],eet.position[1],0.30), eet.orientation))
    tp.move_cartesian(eet, step_len=0.01)
    tp.gripper(False)
    tp.move_cartesian(Pose((eet.position[0],eet.position[1],0.30), eet.orientation))
    print(" placed; seated", tp.env._part_is_seated_in_rack(o.name, tp.state), "steps", len(tp.actions))
print("term", tp.terminated, "steps", len(tp.actions), "time", time.time()-t)
