# Dynamo3DEnv (variable object count)

A 3D navigation-among-movable-objects task where the robot must reach a goal region on the floor while several chair obstacles are placed in the workspace.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Generalization

This environment contains a VARIABLE number of objects. Your program must handle ANY number of them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.

## Observation

Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed index layout. Read it with:

- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` to enumerate objects,
- `state.get(obj, feature)` to read a feature.

| **Type** | **Features** | **Example objects** |
| --- | --- | --- |
| mujoco_movable_object | x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, bb_x, bb_y, bb_z | obstacle_chair0, obstacle_chair1, ... |
| mujoco_tidybot_robot | pos_base_x, pos_base_y, pos_base_rot, pos_arm_joint1, pos_arm_joint2, pos_arm_joint3, pos_arm_joint4, pos_arm_joint5, pos_arm_joint6, pos_arm_joint7, pos_gripper, vel_base_x, vel_base_y, vel_base_rot, vel_arm_joint1, vel_arm_joint2, vel_arm_joint3, vel_arm_joint4, vel_arm_joint5, vel_arm_joint6, vel_arm_joint7, vel_gripper | robot |

The count-defining objects are named `obstacle_chair0`, `obstacle_chair1`, ... and their number changes per episode.

## Action Space

Actions: base pos and yaw (3), arm joints (7), gripper pos (1)

## Reward

The primary reward is for successfully placing objects at their target locations.
- A reward of +1.0 is given for each object placed within a 5cm tolerance of its target.
- A smaller positive reward is given for objects within a 10cm tolerance to guide the robot.
- A small negative reward (-0.01) is applied at each timestep to encourage efficiency.
The episode terminates when all objects are placed at their respective targets.


