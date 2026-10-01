src=open('approach.py').read()
def rep(old,new,cnt=1):
    global src
    assert src.count(old)==cnt, (old, src.count(old))
    src=src.replace(old,new)
rep("""def rect_corners""","""def safe_th(t):
    \"\"\"wrap and keep away from +-pi (float32 +-pi falls outside [-pi, pi]).\"\"\"
    t = wrap(t)
    if abs(t) > math.pi - 1e-4:
        return math.copysign(math.pi - 2e-4, t)
    return t


def rect_corners""")
rep("""            th_t = min(cands, key=lambda t: abs(wrap(t - math.pi / 2)))""","""            th_t = min(cands, key=lambda t: abs(wrap(t - math.pi / 2)))
            blk = wrap(th_t + rel)
            th_t = safe_th(th_t + (safe_th(blk) - blk))""")
src = src.replace("math.atan2(-n[1], -n[0])", "safe_th(math.atan2(-n[1], -n[0]))")
open('approach.py','w').write(src)
