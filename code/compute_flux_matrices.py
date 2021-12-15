#!/bin/python3

#==============================================================================
# libraries
#==============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import datetime
import re
import json

#==============================================================================
# global
#==============================================================================
datadir = Path('data')
if not datadir.is_dir():
    raise ValueError("Data dir doesn'nt exist!")

resdir = Path('results/')
if not resdir.is_dir():
    raise ValueError('No results directory!')

#==============================================================================
# I/O
#==============================================================================
data_list = [f for f in (datadir / 'social_distancing_metrics').glob('**/*') if f.is_file()]
data_list.sort()
nfiles = len(data_list)
print("nfiles = {:d}".format(nfiles))

resfile = resdir / 'safegraph_analysis_tp.hdf5'
complevel=7
complib='zlib'

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
    print(f"Store has {len(store.keys())} entries.")
store.close()

#==============================================================================
# load CBG index
#==============================================================================
key = "/clustering/clusters"
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
    clusters = store[key]

key = "/clustering/cbgs_clusters"
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
    cbgs_labels = store[key]

df = clusters.copy().loc[:,'leaves'].reset_index().set_index('leaves')
cbgs_labels['index'] = -1

for cbgs in cbgs_labels.index:
    cbgs_labels.at[cbgs, 'index'] = df.at[cbgs_labels.at[cbgs, 'leaves'],'index']


#==============================================================================
# construct individual flux matrices
#==============================================================================
np.random.seed(123)
i_list = np.random.permutation(np.arange(len(data_list)))

# offset = 100
# nsel = 100
# i_list = i_list[offset:offset+nsel]

N = len(clusters)
pattern = "\d{4}-\d{2}-\d{2}"
idump = 10000
for ii, i in enumerate(i_list):
    # file name
    res = re.findall(pattern, data_list[i].name.replace('.csv.gz',''))
    if len(res) != 1:
        raise ValueError
    pref = res[0]
    print(f"{ii+1} / {len(i_list)}", i, pref)

    # define all store keys
    key_f = "/fluxes/{:s}".format(pref)
    key_e = "/fluxes_escaped/{:s}".format(pref)

    # if entry already exist, skip
    with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
        if (key_f in store) and (key_e in store):
            print("Files already computed... Skipping!")
            continue

    # load dataframe
    df = pd.read_csv(data_list[i])
    df.set_index('origin_census_block_group', inplace=True)

    flux_mat = pd.DataFrame(np.zeros((N,N), dtype=np.int_), dtype=np.int_, index=clusters.index, columns=clusters.index)
    count_escaped = pd.DataFrame(np.zeros((N,1), dtype=np.int_), index=clusters.index, columns=['escaped_count'])
    population = pd.DataFrame(np.zeros((N,1), dtype=np.int_), index=clusters.index, columns=['population'])

    # remove cbg not in master index
    idx = df.index.intersection(cbgs_labels.index)
    idx = df.index.difference(idx)
    print(f"  removing {len(idx)} / {len(df)} lines.")
    df.drop(index=idx, inplace=True)

    # add to the matrix of fluxes
    for k, c in enumerate(df.index):
        if k % idump == 0:
            print("{:2s}{:d} / {:d}".format("", k, len(df.index)))

        flux_dict = {int(key): int(val) for key,val in json.loads(df.at[c, 'destination_cbgs']).items()}
        n = cbgs_labels.at[c,'index']

        population.loc[n] += df.at[c, 'device_count']

        for c_dest in flux_dict:
            count = flux_dict[c_dest]
            if c_dest in df.index:
                n_dest = cbgs_labels.at[c_dest,'index']
                flux_mat.loc[n,n_dest] += count
            else:
                count_escaped.loc[n] += count

    # diagonal entries are the population in each community
    for n in flux_mat.index:
        flux_mat.at[n,n] = population.at[n, 'population']

    with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
        store[key_f] = flux_mat
        store[key_e] = count_escaped
