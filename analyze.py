#!/usr/bin/env python
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import numpy as nm

import soops.scoop_outputs as so
import soops.ioutils as io
import soops.plot_selected as sps

import sys
if 'sfepy' not in sys.path:
    sys.path.append(op.expanduser('~/projects/sfepy-git'))

import time_tensors as tt

filename = 'output/matrix-dw_laplace-8192-layouts-reports/results.h5'
#filename = 'output/tmp-plot-scatter/results.h5'
output_dir = 'output/tmp'

df = io.get_from_store(filename, 'df')
par_keys = set(io.get_from_store(filename, 'par_keys').to_list())

data = so.init_plugin_data(df, par_keys, output_dir, filename)
data = tt.collect_stats(df, data)
tt.check_rnorms(df, data)
data = tt.setup_uniques(df, data)
# data = tt.select_data(df, data, orders=[1])
data = tt.select_data(df, data, omit_functions=['.*dat.*'])
data = tt.setup_styles(df, data)

ldf = data.ldf
fdf = data.fdf

####### ... that spaths and opt are not 1:1...

mii = pd.MultiIndex.from_arrays([ldf.lib, ldf.opt, ldf.spaths]).unique().sort_values()
ldf.plot.scatter(x='opt', y='spaths')

sdf = (ldf[['lib', 'opt', 'spaths']].drop_duplicates()
       .sort_values(['lib', 'opt', 'spaths'], ignore_index=True))

cat = sdf['lib'].astype('category').cat
sdf.plot.scatter(x='opt', y='spaths', color=cat.codes)


sldf = ldf.sort_values('layout')

cat = sldf['lib'].astype('category').cat

sldf.plot.scatter(x='layout', y='rtwwmean', c=cat.codes, ls='None', marker='o', colormap='viridis')

#######

sldf = ldf.sort_values(['lib', 'rtwwmean'])

select = sps.select_by_keys(ldf, ['lib'])
styles = {'lib' : {'color' : 'viridis'}}
styles = sps.setup_plot_styles(select, styles)
#colors = sps.get_cat_style(select, 'lib', styles, 'color')

fig, ax = plt.subplots()
used = None
for ii, lib in enumerate(data.uniques['lib']):
    sdf = sldf[sldf['lib'] == lib]
    style_kwargs, indices = sps.get_row_style(
        sdf.iloc[0], select, {}, styles
    )
    used = sps.update_used(used, indices)
    ax.plot(sdf['layout'], sdf['rtwwmean'], ls='None', marker='o', alpha=0.6,
            **style_kwargs)

sps.add_legend(ax, select, styles, used)

order = data.orders[0]
fig, ax = plt.subplots()
used = None
for ii, lib in enumerate(data.uniques['lib']):
    sdf = sldf[(sldf['lib'] == lib) &
               (sldf['order'] == order)]
    if not len(sdf): continue

    style_kwargs, indices = sps.get_row_style(
        sdf.iloc[0], select, {}, styles
    )
    used = sps.update_used(used, indices)
    ax.plot(sdf['layout'], sdf['rtwwmean'], ls='None', marker='o', alpha=0.6,
            **style_kwargs)

sps.add_legend(ax, select, styles, used)

#######

# includes libs missing due to timeout/memory requirements as empty rows.

sldf = ldf.sort_values(['lib', 'rtwwmean'])

select = sps.select_by_keys(ldf, ['layout'])
styles = {'layout' : {'color' : 'viridis'}}
styles = sps.setup_plot_styles(select, styles)

order = data.orders[2]
fig, ax = plt.subplots()
used = None
libs = data.uniques['lib']
ax.set_yticks(nm.arange(len(libs)))
ax.set_yticklabels(libs)
ax.grid(True)
for ii, layout in enumerate(data.uniques['layout']):
    sdf = sldf[(sldf['layout'] == layout) &
               (sldf['order'] == order)]
    if not len(sdf): continue

    style_kwargs, indices, used = sps.get_row_style_used(
        sdf.iloc[0], select, {}, styles, used
    )
    if layout == 'cqgvd0':
        style_kwargs['color'] = 'r'
        style_kwargs['zorder'] = 100
    ax.plot(sdf['rtwwmean'], nm.searchsorted(libs, sdf['lib']), ls='None',
            marker='o', alpha=0.6, **style_kwargs)

# sps.add_legend(ax, select, styles, used)

aa

plt.figure()
colors = ldf['lib'].astype('category').cat.codes
#markers = tuple(ldf['lib'].astype('category').values)
ldf.plot.scatter(x='rtwwmean', y='rmmean', c=colors, colorbar=True, colormap='viridis', logx=True)

plt.figure()
colors = ldf['spaths'].astype('category').cat.codes
ldf.plot.scatter(x='rtwwmean', y='rmmean', c=colors, colorbar=True, colormap='viridis', logx=True)

gb1 = ldf.groupby(['lib'])
gb2 = ldf.groupby(['spaths'])
gb3 = ldf.groupby(['variant'])
gb1.groups

plt.figure()
out1 = gb1['rtwwmean'].plot(ls='None', marker='o')
plt.legend()

plt.figure()
out2 = gb2['rtwwmean'].plot(ls='None', marker='o')
plt.legend()

plt.figure()
out3 = gb3['rtwwmean'].plot(ls='None', marker='o')
plt.legend()
