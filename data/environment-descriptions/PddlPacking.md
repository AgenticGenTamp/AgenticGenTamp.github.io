# PR2Packed

A PR2 robot must pick up every block from the table and place it on the green plate. This is the `packed` task and motion planning benchmark: the blocks start scattered on the table at random collision-free poses, the plate is small enough that they have to be packed together, and reaching a block may require driving the base as well as moving the arm.

## Variant

A variable number of blocks, each 0.07m square and 0.10m tall, on a plate 0.27m square. Only the left arm is controllable; the right arm is tucked.

`grasp_active` on the robot is 1.0 while a block is held, and the block's own `grasp_active` marks which one. `grasp_tf` is the pose of the held block in the gripper's tool frame, and is all zeros when nothing is held. Poses are `(x, y, z)` position followed by an `(qx, qy, qz, qw)` quaternion.

## Action Space

`Box(-0.2, 0.2, (11,), float32)` except for index 10, which is bounded by +/-1. Actions are bounded relative base position, rotation, and arm joint positions, plus gripper open/close.

| **Index** | **Description** |
| --- | --- |
| 0 | delta base x |
| 1 | delta base y |
| 2 | delta base rotation |
| 3 | delta joint 1 |
| 4 | delta joint 2 |
| 5 | delta joint 3 |
| 6 | delta joint 4 |
| 7 | delta joint 5 |
| 8 | delta joint 6 |
| 9 | delta joint 7 |
| 10 | gripper open/close |

The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.

**Closing** grasps a block only if the gripper is actually around one: the block's centre must be between the fingers (within 0.035m of the tool frame laterally) and no further than 0.06m along the approach axis, which is where the fingers stop. Hovering above a block does not grasp it, however close the tool frame is -- you have to bring the fingers down around it. The nearest qualifying block is taken, and held rigidly until the gripper opens. Check `grasp_active` to see whether a grasp took.

**Opening** drops the held block straight down onto whichever surface it is above (the plate if it is over the plate, otherwise the table, otherwise the floor). The drop is REFUSED if the block would land overlapping another block or a fixed body: the block stays held and the gripper stays closed, so `grasp_active` remains 1. Move somewhere clear and open again.

Base and arm targets are clipped to the robot's joint limits, except for base rotation (index 2) and the two continuous arm roll joints (indices 7 and 9), which wrap around at +/-pi. Dynamics are kinematic: a motion that would put the robot or the block it is holding in collision with the table, the plate, or another block is rejected, leaving the robot where it was. The gripper command on that same step still applies.

## Reward

-1 per step. The episode terminates when every block rests on the plate AND no two blocks overlap -- the plate is small enough that they have to be packed, so dropping them all at one spot does not count. The return is the negated number of steps taken, so a better policy is a shorter one.


## Variable Object Count

The number of blocks changes between episodes, so observations are object-centric rather than a fixed-length vector: each is an `ObjectCentricState` holding one `robot`, the `table` and `plate` surfaces, and `block0`..`blockN-1`.

Feature names per type:

- `robot`: base_x, base_y, base_rot, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper_opening, grasp_active, grasp_tf_x, grasp_tf_y, grasp_tf_z, grasp_tf_qx, grasp_tf_qy, grasp_tf_qz, grasp_tf_qw
- `surface`: pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, half_extent_x, half_extent_y, half_extent_z
- `block`: pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, grasp_active, half_extent_x, half_extent_y, half_extent_z

Iterate the state's objects rather than indexing fixed offsets, so one program handles any number of blocks.
