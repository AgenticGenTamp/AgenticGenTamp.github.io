"""Small vectorized collision checker for a circular mobile manipulator."""
import numpy as np


class Scene:
    def __init__(self, rectangles, radius=.1, gripper_width=.01,
                 gripper_height=.07):
        self.rectangles = np.asarray(rectangles, dtype=float).reshape((-1, 5))
        self.radius = float(radius)
        self.gripper_width = float(gripper_width)
        self.gripper_height = float(gripper_height)
        self.centers = self.rectangles[:, :2]
        angles = self.rectangles[:, 2]
        self.axes = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        self.perps = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
        self.half = self.rectangles[:, 3:5] * .5
        self.eps = 1.e-10

    def _mask(self, ignore):
        mask = np.ones(len(self.rectangles), dtype=bool)
        if ignore is not None:
            mask[ignore] = False
        return mask

    def _boxes_clear(self, centers, angles, width, height, mask):
        """Batch of candidate OBBs versus fixed scene OBBs and world walls."""
        axis = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        perp = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
        half_width = np.broadcast_to(np.asarray(width) * .5, (len(centers),))[:, None]
        half_height = np.broadcast_to(np.asarray(height) * .5, (len(centers),))[:, None]
        extent = np.abs(axis) * half_width + np.abs(perp) * half_height
        result = np.all(centers - extent >= -self.eps, axis=1)
        result &= np.all(centers + extent <= 2.5 + self.eps, axis=1)
        if not np.any(mask):
            return result
        obs_axis, obs_perp = self.axes[mask], self.perps[mask]
        obs_half = self.half[mask]
        delta = centers[:, None, :] - self.centers[None, mask, :]
        aa = np.abs(axis @ obs_axis.T)
        ab = np.abs(axis @ obs_perp.T)
        ba = np.abs(perp @ obs_axis.T)
        bb = np.abs(perp @ obs_perp.T)
        distance = np.abs(np.einsum('mni,mi->mn', delta, axis))
        separated = distance >= (half_width + aa * obs_half[:, 0] +
                                  ab * obs_half[:, 1] - self.eps)
        distance = np.abs(np.einsum('mni,mi->mn', delta, perp))
        separated |= distance >= (half_height + ba * obs_half[:, 0] +
                                   bb * obs_half[:, 1] - self.eps)
        distance = np.abs(np.einsum('mni,ni->mn', delta, obs_axis))
        separated |= distance >= (obs_half[:, 0] + aa * half_width +
                                   ba * half_height - self.eps)
        distance = np.abs(np.einsum('mni,ni->mn', delta, obs_perp))
        separated |= distance >= (obs_half[:, 1] + ab * half_width +
                                   bb * half_height - self.eps)
        result &= np.all(separated, axis=1)
        return result

    def valid_many(self, configurations, ignore=None, held=None):
        """Check rows [base_x, base_y, theta, arm_joint].

        ``ignore`` is an obstacle index or a sequence of obstacle indices.
        ``held`` is [forward, lateral, relative_angle, width, height], with
        translations expressed in the robot base frame.
        """
        q = np.asarray(configurations, dtype=float).reshape((-1, 4))
        positions, angles = q[:, :2], q[:, 2]
        valid = np.all(np.isfinite(q), axis=1)
        valid &= np.all(positions >= self.radius - self.eps, axis=1)
        valid &= np.all(positions <= 2.5 - self.radius + self.eps, axis=1)
        mask = self._mask(ignore)
        if np.any(mask):
            delta = positions[:, None, :] - self.centers[None, mask, :]
            forward = np.abs(np.einsum('mni,ni->mn', delta, self.axes[mask]))
            lateral = np.abs(np.einsum('mni,ni->mn', delta, self.perps[mask]))
            dx = np.maximum(forward - self.half[mask, 0], 0.)
            dy = np.maximum(lateral - self.half[mask, 1], 0.)
            valid &= np.all(dx * dx + dy * dy >= self.radius ** 2 - self.eps,
                            axis=1)
        direction = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        gripper_center = positions + q[:, 3, None] * direction
        valid &= self._boxes_clear(gripper_center, angles, self.gripper_width,
                                   self.gripper_height, mask)
        # The arm shaft can strike clutter even when base and gripper clear it.
        shaft_length = np.maximum(q[:, 3] - self.radius, 0.)
        shaft_center = positions + ((self.radius + q[:, 3]) * .5)[:, None] * direction
        shaft_clear = self._boxes_clear(shaft_center, angles, shaft_length, .01, mask)
        valid &= shaft_clear | (shaft_length <= self.eps)
        if held is not None:
            held = np.asarray(held, dtype=float)
            perpendicular = np.stack((-direction[:, 1], direction[:, 0]), axis=-1)
            held_center = positions + direction * held[0] + perpendicular * held[1]
            valid &= self._boxes_clear(held_center, angles + held[2], held[3],
                                       held[4], mask)
        return valid

    def valid(self, q, ignore=None, held=None):
        return bool(self.valid_many(np.asarray(q).reshape(1, 4), ignore, held)[0])
