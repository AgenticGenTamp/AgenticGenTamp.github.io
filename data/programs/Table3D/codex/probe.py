import argparse
import numpy as np
from env_client import make_env


def dump(state):
    rows = []
    for name in sorted(state.get_object_names()):
        obj = state.get_object_from_name(name)
        feats = state.type_features[obj.type]
        vals = {f: state.get(obj, f) for f in feats}
        rows.append((name, obj.type.name, vals))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action", type=int, default=-1)
    ap.add_argument("--value", type=float, default=0.4)
    args = ap.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed)
    print("max_steps", env.max_steps, "info", info)
    for row in dump(state): print(row)
    if args.action >= 0:
        a = np.zeros(11, dtype=np.float32)
        a[args.action] = args.value
        nxt, rew, term, trunc, info = env.step(a)
        print("RESULT", rew, term, trunc, info)
        before = {n: v for n, _, v in dump(state)}
        for n, typ, vals in dump(nxt):
            changes = {k: round(vals[k] - before[n][k], 6) for k in vals if vals[k] != before[n][k]}
            if changes: print("CHANGE", n, changes)
    env.close()


if __name__ == "__main__": main()
