# PR2Blocked

A PR2 robot must get **any one green block** onto the green plate. This is the `blocked` task and motion planning benchmark. One green block sits on the near table inside a pen of three fixed walls, and a red block stands in the pen's one gap. The walls are as tall as the block, so the only grasp that clears them is a horizontal one -- and the red block is in the way of it. Either move the red block aside and take the penned block, or fetch a spare green block from the far table.

The red block is movable and graspable, but it is not a green block: putting it on the plate does not finish the episode.

## Variant

Every block is 0.07m square and 0.14m tall, on a plate 0.60m square. The pen leaves a 0.15m gap, measured centre to centre, between the penned block and the blocker. There may be any number of spare green blocks on the far table, including none. The two tables are nine metres apart, so fetching a spare is a long base drive. Only the left arm is controllable; the right arm is tucked.

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

**Closing** grasps a block only if the gripper is actually around one: the block's centre must be within 0.035m of the tool frame across the approach axis and 0.070m along the gripper's vertical, and no further than 0.045m along the approach axis itself, which is where the fingers stop. That is half a block *width*, not half its height: this is a side grasp, so you have to bring the fingers in from the side rather than down from above. The nearest qualifying block is taken, and held rigidly until the gripper opens. Check `grasp_active` to see whether a grasp took.

**Opening** drops the held block straight down onto whichever surface it is above (the plate if it is over the plate, otherwise whichever table, otherwise the floor). The drop is REFUSED if the block would land overlapping another block or a fixed body: the block stays held and the gripper stays closed, so `grasp_active` remains 1. Move somewhere clear and open again.

Base and arm targets are clipped to the robot's joint limits, except for base rotation (index 2) and the two continuous arm roll joints (indices 7 and 9), which wrap around at +/-pi. Dynamics are kinematic: a motion that would put the robot or the block it is holding in collision with a table, the plate, a wall, or another block is rejected, leaving the robot where it was. The gripper command on that same step still applies.

## Reward

-1 per step. The episode terminates as soon as any green block rests on the plate. The return is the negated number of steps taken, so a better policy is a shorter one -- which is the choice the task poses: moving the blocker is a short manipulation, and fetching a spare is a long drive.


## Variable Object Count

The number of spare green blocks changes between episodes, and may be zero, so observations are object-centric rather than a fixed-length vector: each is an `ObjectCentricState` holding one `robot`, the `near_table`, `far_table` and `plate` surfaces, `green0` (the penned block, always present) with `green1`..`greenN` for any spares, and `blocker`.

Any one green block on the plate ends the episode, so spares are alternatives rather than extra work. When there are none, the penned block is the only green one and the blocker has to be moved.

Feature names per type:

- `robot`: base_x, base_y, base_rot, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper_opening, grasp_active, grasp_tf_x, grasp_tf_y, grasp_tf_z, grasp_tf_qx, grasp_tf_qy, grasp_tf_qz, grasp_tf_qw
- `surface`: pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, half_extent_x, half_extent_y, half_extent_z
- `block`: pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, grasp_active, half_extent_x, half_extent_y, half_extent_z

Iterate the state's objects rather than indexing fixed offsets, so one program handles any number of spares.
