#!/bin/python3

#-------------------------------------------------------------------------------
# imports
#-------------------------------------------------------------------------------
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import sys
import h5py
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.append('../code/')
from functions import get_residual_susceptible as get_final_s
from functions import lattice_2d_ode as func_ode
from functions import lattice_2d_event_upperbound as func_event_upperbound

#-------------------------------------------------------------------------------
# parameters
#-------------------------------------------------------------------------------
N1 = 1024     # number of lattice sites
N2 = 2**6     # number of lattice sites
# N = 16     # number of lattice sites
T = 10000.     # final time
gamma_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
# gamma_list = [0.8, 0.9, 0.95, 0.98]
tdump = 1.e-1
# tdump = 5.e-1
method='DOP853'
resfile = '../results/lattice_2d_nneighbors.hdf5'
pathpref = Path('N{:d}_M{:d}'.format(N1,N2))
complevel=7
complib='zlib'

#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  for gamma in gamma_list:
    tstamp=(datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")
    print(tstamp, "Starting simulations gamma={:.2e}".format(gamma))
    sinf = get_final_s(gamma)
    # jlo = 1.
    # jhi = 0.
    jlo = 0.
    jhi = 0.
    print("s(inf) = {:.2e}".format(sinf))

    J = np.zeros((N1,N2))
    if N2 == 1.:
      J[0,0]=1.
    else:
      i0 = N2//2-1
      J[0,i0]=1.
      J[0,i0+1]=1.

    S = 1. - J
    X = np.ravel(np.array([S, J]), order='C')

    event_upperbound = func_event_upperbound
    event_upperbound.terminal = True

    sol = solve_ivp(func_ode, t_span=[0., T], y0=X, method=method, \
            args = [gamma, N1, N2, jlo, jhi], t_eval = np.linspace(0., T, int(T/tdump)+1), \
            events=event_upperbound)

    times = sol.t
    SJs = np.array([np.reshape(x, (2,N1,N2), order='C') for x in sol.y.T])
    Ss = SJs[:,0]
    Js = SJs[:,1]

    # print(np.max(Js[-1].reshape(-1,2**5, 2**5), axis=2))
    # print(np.max(Ss[-1].reshape(-1,2**5, 2**5), axis=2))
    # print(np.mean(Js[-1,-1]))
    print(Ss.shape, Js.shape)

    # save
    path = str(pathpref / 'gamma_{:.2e}'.format(gamma))
    with h5py.File(resfile,'a') as f5py:
        if not (path in f5py.keys()):
            grp = f5py.create_group(path)
        grp = f5py[path]

        name = "times"
        if name in grp.keys():
            del grp[name]
        dset = grp.create_dataset(name, shape=times.shape, dtype=times.dtype, data=times, \
                          compression="gzip", compression_opts=complevel)

        name = "susceptible"
        if name in grp.keys():
            del grp[name]
        dset = grp.create_dataset(name, shape=Ss.shape, dtype=Ss.dtype, data=Ss, \
                          compression="gzip", compression_opts=complevel)

        name = "infected"
        if name in grp.keys():
            del grp[name]
        dset = grp.create_dataset(name, shape=Js.shape, dtype=Js.dtype, data=Js, \
                          compression="gzip", compression_opts=complevel)
    print("data written to {:s}>{:s}".format(str(resfile), path))

