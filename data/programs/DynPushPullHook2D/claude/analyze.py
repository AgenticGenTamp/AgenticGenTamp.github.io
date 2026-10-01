"""Analyse a rendered environment PNG: colors + connected components."""

import sys
from collections import defaultdict

import numpy as np

from pngread import read_png


# ---------------------------------------------------------------- labeling
def label_mask(mask):
    """Run-based 4-connected labeling. Returns (labels, count) with 0 = bg."""
    h, w = mask.shape
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask
    diff = np.diff(padded.astype(np.int8), axis=1)

    rows = []          # per row: list of (x0, x1_exclusive, label)
    prev = []
    for y in range(h):
        d = diff[y]
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        cur = []
        pi = 0
        for x0, x1 in zip(starts, ends):
            lab = 0
            while pi < len(prev) and prev[pi][1] <= x0:
                pi += 1
            j = pi
            while j < len(prev) and prev[j][0] < x1:
                if lab == 0:
                    lab = find(prev[j][2])
                else:
                    union(lab, prev[j][2])
                    lab = find(lab)
                j += 1
            if lab == 0:
                lab = len(parent)
                parent.append(lab)
            cur.append((int(x0), int(x1), lab))
        rows.append(cur)
        prev = cur

    remap = {}
    out = []
    for y, cur in enumerate(rows):
        for x0, x1, lab in cur:
            r = find(lab)
            if r not in remap:
                remap[r] = len(remap) + 1
            out.append((y, x0, x1, remap[r]))
    return out, len(remap)


def component_stats(mask, min_pixels=1):
    """Return list of dicts with bbox/centroid/area for each component."""
    runs, n = label_mask(mask)
    acc = defaultdict(lambda: dict(area=0, sx=0.0, sy=0.0,
                                   x0=10**9, x1=-1, y0=10**9, y1=-1))
    for y, x0, x1, lab in runs:
        a = acc[lab]
        cnt = x1 - x0
        a["area"] += cnt
        a["sx"] += (x0 + x1 - 1) * cnt / 2.0
        a["sy"] += y * cnt
        a["x0"] = min(a["x0"], x0)
        a["x1"] = max(a["x1"], x1 - 1)
        a["y0"] = min(a["y0"], y)
        a["y1"] = max(a["y1"], y)
    comps = []
    for lab, a in acc.items():
        if a["area"] < min_pixels:
            continue
        comps.append(dict(area=a["area"],
                          bbox=(a["x0"], a["x1"], a["y0"], a["y1"]),
                          centroid=(round(a["sx"] / a["area"], 1),
                                    round(a["sy"] / a["area"], 1))))
    comps.sort(key=lambda c: -c["area"])
    return comps


# ---------------------------------------------------------------- helpers
def color_key(img):
    """Pack HxWx3 into a single int32 code per pixel."""
    return (img[:, :, 0].astype(np.int32) << 16
            | img[:, :, 1].astype(np.int32) << 8
            | img[:, :, 2].astype(np.int32))


def main(path):
    img = read_png(path)
    h, w, _ = img.shape
    print("image size: %d x %d (W x H)" % (w, h))

    codes = color_key(img)
    vals, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)
    print("\ndistinct colors: %d  (top 12 by pixel count)" % len(vals))
    for i in order[:12]:
        c = int(vals[i])
        print("  RGB(%3d,%3d,%3d)  %8d px" %
              ((c >> 16) & 255, (c >> 8) & 255, c & 255, counts[i]))

    bg = int(vals[order[0]])
    bg_rgb = ((bg >> 16) & 255, (bg >> 8) & 255, bg & 255)
    print("\nbackground color assumed: RGB%s" % (bg_rgb,))

    print("\n=== connected components per non-background color "
          "(>=%d px) ===" % MINPX)
    for i in order:
        c = int(vals[i])
        if c == bg or counts[i] < MINPX:
            continue
        rgb = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
        mask = codes == c
        comps = component_stats(mask, min_pixels=MINPX)
        if not comps:
            continue
        print("\ncolor RGB%-16s total %d px, %d component(s) >=%d px"
              % (str(rgb), counts[i], len(comps), MINPX))
        for k, cp in enumerate(comps):
            x0, x1, y0, y1 = cp["bbox"]
            print("   comp %-2d area=%-7d bbox x[%d..%d] y[%d..%d] "
                  "(w=%d,h=%d) centroid=(%.1f,%.1f)"
                  % (k, cp["area"], x0, x1, y0, y1,
                     x1 - x0 + 1, y1 - y0 + 1,
                     cp["centroid"][0], cp["centroid"][1]))

        # for the biggest component, look for an enclosed hole (frame inner box)
        if len(comps) and comps[0]["area"] > 5000:
            x0, x1, y0, y1 = comps[0]["bbox"]
            if (x1 - x0 + 1) > w * 0.5 and (y1 - y0 + 1) > h * 0.5:
                sub = ~mask[y0:y1 + 1, x0:x1 + 1]
                holes = component_stats(sub, min_pixels=50)
                sh, sw = sub.shape
                for hc in holes:
                    hx0, hx1, hy0, hy1 = hc["bbox"]
                    if hx0 > 0 and hy0 > 0 and hx1 < sw - 1 and hy1 < sh - 1:
                        print("   -> enclosed hole (inner bbox): "
                              "x[%d..%d] y[%d..%d] (w=%d,h=%d) area=%d"
                              % (hx0 + x0, hx1 + x0, hy0 + y0, hy1 + y0,
                                 hx1 - hx0 + 1, hy1 - hy0 + 1, hc["area"]))


MINPX = 30

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "mcp_renders/state_seed42_init42.png"
    if len(sys.argv) > 2:
        MINPX = int(sys.argv[2])
    main(p)
