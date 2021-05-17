import test
import sys
from getopt import getopt

import models

opts = getopt(sys.argv[1:], 'm:')[0]

model_name = opts[0][1] if len(opts) else None

test.run_tests(model_name)