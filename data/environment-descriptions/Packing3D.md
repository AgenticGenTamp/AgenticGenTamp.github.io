# Packing3DEnv (variable object count)

A 3D packing environment where the goal is to place a set of parts into a rack without collisions.

The robot is a Kinova Gen-3 with 7 degrees of freedom that can grasp and manipulate objects. The environment consists of:
- A **table** with dimensions 0.400m × 0.800m × 0.500m
- A **rack** (purple) with half-extents (0.1, 0.15, 0.02)
- **Parts** (green) that must be packed into the rack. Parts are sampled with half-extents in (0.05, 0.05, 0.01, 0) to (0.05, 0.05, 0.01, 0) and a probability 0.5 of being triangle-shaped (triangles are represented as triangular prisms with depth 0.020m when used).


## Generalization

This environment contains a VARIABLE number of objects. Your program must handle ANY number of them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.

## Observation

Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed index layout. Read it with:

- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` to enumerate objects,
- `state.get(obj, feature)` to read a feature.

| **Type** | **Features** | **Example objects** |
| --- | --- | --- |
| Kinematic3DCuboid | pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, grasp_active, object_type, half_extent_x, half_extent_y, half_extent_z | rack, part0, part1, ... |
| Kinematic3DRobot | pos_base_x, pos_base_y, pos_base_rot, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, finger_state, grasp_active, grasp_tf_x, grasp_tf_y, grasp_tf_z, grasp_tf_qx, grasp_tf_qy, grasp_tf_qz, grasp_tf_qw | robot |
| Kinematic3DTriangle | pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, grasp_active, triangle_type, side_a, side_b, depth | part0, part1, ... |

The count-defining objects are named `part0`, `part1`, ... and their number changes per episode.

## Action Space

An action space for mobile manipulation with a 7 DOF robot that can open and close its gripper.

Actions are bounded relative base position, rotation, and joint positions, and open / close.

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


## Reward

The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when all parts are placed in the rack and none are grasped

The goal is considered reached when:
1. The robot is not currently grasping any part
2. Every part is resting on (supported by) the rack surface

Support is determined based on contact between a part and the rack within a small distance threshold (configured by the environment).

This encourages the robot to efficiently pack the parts into the rack while avoiding infinite episodes.


