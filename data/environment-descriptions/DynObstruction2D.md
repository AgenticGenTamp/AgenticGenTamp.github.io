# DynObstruction2DEnv (variable object count)

A 2D physics-based environment where the goal is to place a target block onto a target surface using a two-fingered robot with PyMunk physics simulation. The block must be completely on the surface. The target surface may be initially obstructed.

The robot has a movable circular base and an extendable arm with gripper fingers. Objects can be grasped and released through gripper actions. All objects follow realistic physics including gravity, friction, and collisions.

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.


## Generalization

This environment contains a VARIABLE number of objects. Your program must handle ANY number of them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.

## Observation

Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed index layout. Read it with:

- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` to enumerate objects,
- `state.get(obj, feature)` to read a feature.

| **Type** | **Features** | **Example objects** |
| --- | --- | --- |
| dyn_rectangle | x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, width, height, mass | obstruction0, obstruction1, ... |
| kin_robot | x, y, theta, vx_base, vy_base, omega_base, vx_arm, vy_arm, omega_arm, vx_gripper_l, vy_gripper_l, omega_gripper_l, vx_gripper_r, vy_gripper_r, omega_gripper_r, static, base_radius, arm_joint, arm_length, gripper_base_width, gripper_base_height, finger_gap, finger_height, finger_width | robot |
| target_block | x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, width, height, mass | target_block |
| target_surface | x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, width, height | target_surface |

The count-defining objects are named `obstruction0`, `obstruction1`, ... and their number changes per episode.

## Action Space

The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | dgripper | Change in gripper gap (positive is open) | -0.020 | 0.020 |


## Reward

A penalty of -1.0 is given at every time step until termination, which occurs when the target block is completely "on" the target surface. The "on" condition requires that the bottom vertices of the target block are within the bounds of the target surface.


