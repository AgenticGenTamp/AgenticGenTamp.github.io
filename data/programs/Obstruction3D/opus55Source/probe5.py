from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
sim = ObjectCentricObstruction3DEnv(num_obstructions=1, allow_state_access=True)
a=sim.robot.arm
print(a.joint_lower_limits[:7], a.joint_upper_limits[:7])
print(a.get_joint_positions())
print(a.closed_fingers_state, a.open_fingers_state)
print(sim.config.max_collision_check_step)
