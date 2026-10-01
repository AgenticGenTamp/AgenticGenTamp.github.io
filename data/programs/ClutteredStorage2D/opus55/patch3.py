src = open('approach.py').read()
old = "                score = -space * 10 + math.hypot(gx - self.rx, gy - self.ry) * 0.1"
new = "                score = -space * 2 + max(abs(gx - self.rx), abs(gy - self.ry))"
assert old in src
src = src.replace(old, new)
open('approach.py','w').write(src)
