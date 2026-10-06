"""World-frame polygons read directly from the pymunk space of the scoop-pour env."""
import pymunk


def _inner(env):
    """Find the object-centric env holding .robot and .pymunk_space."""
    e = env
    for _ in range(20):
        if hasattr(e, "robot") and hasattr(e, "pymunk_space") and e.pymunk_space is not None:
            return e
        for attr in ("_current_backend", "_object_centric_env", "env", "_env", "unwrapped", "_current_env", "inner_env"):
            nxt = getattr(e, attr, None)
            if nxt is not None and nxt is not e:
                e = nxt
                break
        else:
            break
    # fallback: scan attributes
    for v in vars(e).values():
        if hasattr(v, "robot") and hasattr(v, "pymunk_space"):
            return v
    raise RuntimeError("could not locate pymunk env")


def _world(shape):
    b = shape.body
    return [tuple(b.local_to_world(v)) for v in shape.get_vertices()]


def shapes(env):
    """Return dict: left_finger, right_finger, gripper_base, arm, hook (list of rect polys)."""
    e = _inner(env)
    r = e.robot
    out = {
        "left_finger": _world(r._left_finger_shape),   # FINGER collision type
        "right_finger": _world(r._right_finger_shape),  # ARM collision type
        "gripper_base": _world(r._gripper_base_shape),
        "arm": _world(r._arm_shape),
    }
    hook_body = None
    for obj, body in e._state_obj_to_pymunk_body.items():
        if obj.name == "hook":
            hook_body = body
    if hook_body is None or hook_body not in e.pymunk_space.bodies:
        # fall back to held objects
        if r.held_objects:
            hook_body = r.held_objects[0][0][0]
    hs = [s for s in e.pymunk_space.shapes if s.body is hook_body and isinstance(s, pymunk.Poly)]
    out["hook"] = [_world(s) for s in hs]
    return out


if __name__ == "__main__":
    from common import make_env
    env = make_env()
    env.reset(seed=115, options={"object_count": 1})
    for k, v in shapes(env).items():
        if k == "hook":
            for i, p in enumerate(v):
                print(f"hook[{i}]:", [(round(x, 4), round(y, 4)) for x, y in p])
        else:
            print(f"{k}:", [(round(x, 4), round(y, 4)) for x, y in v])
