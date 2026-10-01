import numpy as np, fk, sys
from env_client import make_env
from lib_util import robot, step_to
from expB_common import *
exec(open('expB_q3d.py').read().split("env=make_env(); obs,g")[0].split("import numpy")[1].join(["import numpy",""])) if False else None
