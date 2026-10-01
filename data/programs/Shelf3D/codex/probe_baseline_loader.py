"""Load the known-working first-grasp policy revision for isolated probes."""

import subprocess


namespace = {}
source = subprocess.check_output(
    ["git", "show", "34f37a3:approach.py"], text=True)
exec(compile(source, "approach.py@34f37a3", "exec"), namespace)
GeneratedApproach = namespace["GeneratedApproach"]
