import sys, logging
logging.disable(logging.WARNING)
import approach
orig = approach.Sim.restore_snapshot
def rs(self, snap):
    print("restore best len", len(snap[1]), "eval_m", self.eval_m)
    return orig(self, snap)
approach.Sim.restore_snapshot = rs
osnap = approach.Sim.snapshot
def sn(self, extra=None):
    r = osnap(self, extra); print("  snap len", len(r[1])); return r
approach.Sim.snapshot = sn
exec(open('dbg.py').read().replace('from approach import GeneratedApproach','GeneratedApproach=approach.GeneratedApproach'))
