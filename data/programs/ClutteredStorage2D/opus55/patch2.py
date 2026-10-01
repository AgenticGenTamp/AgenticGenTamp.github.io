src = open('approach.py').read()
old = """                    cost = D * 3 + abs(off) + math.hypot(p[0] - self.rx, p[1] - self.ry) * 0.5
                    opts.append((cost, p, math.atan2(-n[1], -n[0]), D))
                    break"""
new = """                    sgx = min(max(p[0], self.sx1 + 0.15), self.sx1 + self.sw1 - 0.15)
                    cost = (max(abs(p[0] - self.rx), abs(p[1] - self.ry))
                            + max(abs(p[0] - sgx), abs(p[1] - 2.3))) / 0.05 \\
                        + 2 * math.ceil((D - 0.23) / 0.1) + abs(off) * 50
                    opts.append((cost, p, math.atan2(-n[1], -n[0]), D))"""
assert old in src
src = src.replace(old, new)
open('approach.py','w').write(src)
