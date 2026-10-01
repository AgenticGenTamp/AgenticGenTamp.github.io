src=open('approach.py').read()
def rep(old,new):
    global src
    assert src.count(old)==1, old
    src=src.replace(old,new)
rep("""            gain = k - cap
            cost = -gain * 100 + abs(cx - self.rx) / 0.05""","""            gain = k - cap
            trips = sorted(max(abs(bo['center'][0] - cx), abs(2.3 - bo['center'][1]))
                           for bo in self.blocks.values() if not self._inside(bo))
            trip = 2 * sum(trips[:gain]) / 0.05
            cost = -gain * 100 + abs(cx - self.rx) / 0.05 + trip""")
open('approach.py','w').write(src)
