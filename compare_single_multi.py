import os
import sys

import pandas as pd

args = sys.argv[1:]

sdf = pd.read_hdf(args[0]).set_index(['n_cell', 'order'])
mdf = pd.read_hdf(args[1]).set_index(['n_cell', 'order'])

keys = mdf.keys()[::2]

df = mdf[keys] / sdf[keys]

print(df)
print(df.min())

from soops import shell; shell()
