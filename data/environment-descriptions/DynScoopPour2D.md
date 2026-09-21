# CountParameterizedDynScoopPour2DEnv (variable object count)

A 2D physics-based environment where the goal is to move small objects from the left side of a middle wall to the right side. The middle wall is half the height of the world.

The robot has a movable circular base and an extendable arm with gripper fingers. An L-shaped hook is present as a kinematic object that can be grasped. Small objects are dynamic and follow PyMunk physics, but they cannot be grasped directly by the robot.

All objects include physics properties like mass, moment of inertia, and color information for rendering.


## Generalization

This environment contains a VARIABLE number of objects. Your program must handle ANY number of them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.

## Observation

Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed index layout. Read it with:

- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` to enumerate objects,
- `state.get(obj, feature)` to read a feature.

| **Type** | **Features** | **Example objects** |
| --- | --- | --- |
| hook | x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, width, length_side1, length_side2, mass | hook |
| kin_robot | x, y, theta, vx_base, vy_base, omega_base, vx_arm, vy_arm, omega_arm, vx_gripper_l, vy_gripper_l, omega_gripper_l, vx_gripper_r, vy_gripper_r, omega_gripper_r, static, base_radius, arm_joint, arm_length, gripper_base_width, gripper_base_height, finger_gap, finger_height, finger_width | robot |
| small_circle | x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, radius, mass | small_0, small_1, ... |
| small_square | x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, size, mass | small_0, small_1, ... |

The count-defining objects are named `small_0`, `small_1`, ... and their number changes per episode.

## Action Space

The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.030 | 0.030 |
| 1 | dy | Change in robot y position (positive is up) | -0.030 | 0.030 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.098 | 0.098 |
| 3 | darm | Change in robot arm length (positive is out) | -0.080 | 0.080 |
| 4 | dgripper | Change in gripper gap (positive is open) | -0.015 | 0.015 |


## Reward

A penalty of -1.0 is given at every time step until termination, which occurs when at least 50% of the small objects have been moved to the right side of the middle wall.

