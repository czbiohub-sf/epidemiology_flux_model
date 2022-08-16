from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar, Bounds, bracket
import scipy.stats as sst

import matplotlib.pyplot as plt
import matplotlib.colors as mco
import matplotlib.gridspec as mgs
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib import animation
plt.rcParams['svg.fonttype'] = 'none'

sys.path.append(str(Path('..') / 'code'))

from functions import get_infectivity_matrix, func_sir_dX, sir_SI_to_X, sir_X_to_SI, func_sir_dV, guess_scale

resdir = Path('../results/')
if not resdir.is_dir():
  raise ValueError('No results directory!')

resfile = resdir / 'safegraph_analysis.hdf5'
complevel=7
complib='zlib'
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  print(f"File {resfile.stem} has {len(store.keys())} entries.")

figdir = Path('../figures') / '7-LSQ_fit_scales'
if not figdir.is_dir():
  figdir.mkdir(parents=True, exist_ok=True)

colors = ['darkblue', 'darkgreen', 'orange']
labels = ["safegraph", "uniform", "optimal"]

gamma = 1/10.
ti = '2020-03-01'
# tf = '2020-09-01'
tf = '2021-02-16'

tfmt = '%Y-%m-%d'
ti = datetime.datetime.strptime(ti, tfmt)
tf = datetime.datetime.strptime(tf, tfmt)

exts = ['.png', '.svg']

key = "/clustering/clusters"
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  clusters = store[key]
# clusters = pd.read_hdf(resfile, key)
N = len(clusters)
print(f"N = {N}")

population = clusters['population'].to_numpy()
population_inv = np.zeros(population.shape, dtype=np.float_)
idx = population > 0.
population_inv[idx] = 1./population[idx]

path = '/clustering/cssegi'
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  df_cssegi = store[path]

times = df_cssegi.index
idx = (times >= ti) & (times <= tf)
df_cssegi.drop(index=times[~idx], inplace=True)
times = df_cssegi.index.to_pydatetime().tolist()

omega_real = df_cssegi.to_numpy().astype('float64')
domega_real = np.diff(omega_real, axis=0)
domega_real = np.concatenate([omega_real[0].reshape(1,-1), domega_real], axis=0)

# compute the real epidemic sizes per community through time
Ts_real = np.einsum('ta,a->ta', omega_real, population_inv)
Ss_real = 1. - Ts_real
dTs_real = np.einsum('ta,a->ta', domega_real, population_inv)
T_tot_real = np.einsum('ta,a->t', Ts_real, population) / np.sum(population)
S_tot_real = np.einsum('ta,a->t', Ss_real, population) / np.sum(population)
dT_tot_real = np.einsum('ta,a->t', dTs_real, population) / np.sum(population)
Ss_real[~(Ss_real > 0.)] = np.finfo(np.float64).resolution

X = clusters.index.to_numpy()
Y = np.einsum('ta,a->ta', Ts_real, population)[0]
idx = Y > 0

Vs_real = -np.log(Ss_real/Ss_real[0])

# load distances
# dc = 200
dc = 400
# dc = 800
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  distances = store['/clustering/distances'].to_numpy().astype('float64')

#================================================================================
## Integrate the dynamics based on the (S,I) variables
#================================================================================
def fit_scales_si(nt, L, population, Ss_real, g, dt=1., method_solver='DOP853'):
  N = L.shape[0]
  Ss = np.zeros((nt, N), dtype=np.float_)
  Is = np.zeros((nt, N), dtype=np.float_)
  Vs = np.zeros((nt, N), dtype=np.float_)
  scales = np.zeros(nt-1, dtype=np.float_)
  ferrs = np.zeros(nt-1, dtype=np.float_)

  Vs_r = -np.log(Ss_real/Ss_real[0])

  # initialization
  Ss[0] = Ss_real[0]
  Is[0] = 1. - Ss[0]
  Xi = sir_SI_to_X(Ss[0], Is[0])
  X = Xi.copy()
  x1 = 0.

  for i in range(1, nt):
    print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "{:d} / {:d}".format(i, len(times)), np.all(np.isfinite(X)))
    # ODE solver
    func_min = lambda x: np.sum(((sir_X_to_SI(solve_ivp(func_sir_dX, y0=X, t_span=(0,dt), t_eval=[0, dt], \
        args=(L*x, gamma), method=method_solver).y[:,-1],N)[0] - Ss_real[i])*population)**2)

    # guess for scale
    x0 = x1
    x1 = guess_scale(Vs[i-1], Vs_r[i], L, g, Ss_real[0], dt=dt)
    xa, xb, xc, fa, fb, fc, funccalls = bracket(func_min, x0, x1)

    # minimization
    sol = minimize_scalar(func_min, bracket=(xa, xb, xc), bounds=(0., None), method='Brent', options={'maxiter': 100})

    scales[i-1] = sol.x
    ferrs[i-1] = sol.fun

    X = solve_ivp(func_sir_dX, y0=X, t_span=(0,dt), t_eval=[0, dt], \
        args=(L*scales[i-1], g), method=method_solver).y[:,-1]

    Ss[i], Is[i] = sir_X_to_SI(X, N)
    Vs[i] = -np.log(Ss[i]/Ss[0])

  return scales, Ss, Is, Vs

#--------------------------------------------------------------------------------
# SafeGraph infectivity matrix
#--------------------------------------------------------------------------------

### first check that all times have an associated flux matrix
time_fluxes = []
# read the mean flux matrix
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  for rt, dirs, files in store.walk('/fluxes'):
    for f in files:
      t = datetime.datetime.strptime(f, tfmt)
      time_fluxes.append(t)

time_fluxes.sort()

for t in times:
  if not t in time_fluxes:
    raise ValueError("Some dates don't have an associated flux matrix")

N = len(clusters)
L = np.zeros((N,N), dtype=np.float_)
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  for t in times:
    mykey = Path('/fluxes') / t.strftime(tfmt)
    mykey = str(mykey)
    df_flux = store[mykey]
    L += df_flux.to_numpy().astype('float64')
L /= len(times)

L = get_infectivity_matrix(L)
# L = get_infectivity_matrix(L, vscales = 1./clusters.loc[clusters.index, 'area'].to_numpy())

L_sg = np.copy(L)

scales_si_sg, Ss_si_sg, Is_si_sg, Vs_si_sg = fit_scales_si(len(times), L_sg, population, Ss_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_sg, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'sifit_safegraph' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_si_sg, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_si_sg, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_si_sg, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_si_sg, columns=['scale'])

#--------------------------------------------------------------------------------
# uniform infectivity matrix
#--------------------------------------------------------------------------------
N = len(clusters)
L_unif = np.ones((N,N), dtype=np.float_)

scales_si_unif, Ss_si_unif, Is_si_unif, Vs_si_unif = fit_scales_si(len(times), L_unif, population, Ss_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_uniform' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_unif, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'sifit_uniform' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_si_unif, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_si_unif, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_si_unif, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_si_unif, columns=['scale'])

#--------------------------------------------------------------------------------
# optimal infectivity matrix
#--------------------------------------------------------------------------------

# read the optimal matrices and average them
N = len(clusters)
L = np.zeros((N,N), dtype=np.float_)
Nt = len(times)-1
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  # for i in range(len(times)-1):
  for i in range(Nt):
    t = times[i]
    mykey = Path('/fit') / 'betamat_sifit' / 'infectivity_matrices' / t.strftime(tfmt)
    mykey = str(mykey)
    L += store[mykey].to_numpy().astype('float64')
L /= Nt

L_opt = np.copy(L)

scales_si_opt, Ss_si_opt, Is_si_opt, Vs_si_opt = fit_scales_si(len(times), L_opt, population, Ss_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_optimal' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_opt, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'sifit_optimal' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_si_opt, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_si_opt, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_si_opt, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_si_opt, columns=['scale'])

#--------------------------------------------------------------------------------
# distance truncated infectivity matrix
#--------------------------------------------------------------------------------

phi = np.int_(distances < dc*1.0e3)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_sg = store[mykey].to_numpy().astype('float64')

L_trunc = L_sg * phi

scales_si_trunc, Ss_si_trunc, Is_si_trunc, Vs_si_trunc = fit_scales_si(len(times), L_trunc, population, Ss_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_distance_trunc_{:.0f}km'.format(dc) / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_trunc, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'sifit_distance_trunc_{:.0f}km'.format(dc) / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_si_trunc, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_si_trunc, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_si_trunc, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_si_trunc, columns=['scale'])

#================================================================================
## Integrate the dynamics based on the nu variables
#================================================================================

def fit_scales_nu(nt, L, Si, Vs_real, g, dt=1., method_solver='DOP853'):
  N = L.shape[0]
  Vs = np.zeros((nt, N), dtype=np.float_)
  scales = np.zeros(nt-1, dtype=np.float_)
  ferrs = np.zeros(nt-1, dtype=np.float_)

  # initialization
  Vs[0] = Vs_real[0]
  x1 = 0.

  for i in range(1, nt):
    print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "{:d} / {:d}".format(i, len(times)), np.all(np.isfinite(X)))

    # ODE solver
    func_min = lambda x: np.sum((solve_ivp(func_sir_dV, y0=Vs[i-1], t_span=(0,dt), t_eval=[0, dt], \
        args=(L*x, g, Si), \
        method=method_solver).y[:,-1] - Vs_real[i])**2)

    # guess for scale
    x0 = x1
    x1 = guess_scale(Vs[i-1], Vs_real[i], L, g, Si, dt=dt)
    xa, xb, xc, fa, fb, fc, funccalls = bracket(func_min, x0, x1)

    # minimization
    sol = minimize_scalar(func_min, bracket=(xa, xb, xc), bounds=(0., None), method='Brent', options={'maxiter': 100})

    scales[i-1] = sol.x
    ferrs[i-1] = sol.fun

    Vs[i] = solve_ivp(func_sir_dV, y0=Vs[i-1], t_span=(0,dt), t_eval=[0, dt], \
        args=(L*scales[i-1], g, Si), \
        method=method_solver).y[:,-1]

    Ss = np.einsum('ta,a->ta', np.exp(-Vs), Si)

  return scales, Ss, Vs

#--------------------------------------------------------------------------------
# SafeGraph infectivity matrix
#--------------------------------------------------------------------------------

### first check that all times have an associated flux matrix
time_fluxes = []
# read the mean flux matrix
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  for rt, dirs, files in store.walk('/fluxes'):
    for f in files:
      t = datetime.datetime.strptime(f, tfmt)
      time_fluxes.append(t)

time_fluxes.sort()

for t in times:
  if not t in time_fluxes:
    raise ValueError("Some dates don't have an associated flux matrix")

N = len(clusters)
L = np.zeros((N,N), dtype=np.float_)
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  for t in times:
    mykey = Path('/fluxes') / t.strftime(tfmt)
    mykey = str(mykey)
    df_flux = store[mykey]
    L += df_flux.to_numpy().astype('float64')
L /= len(times)

L = get_infectivity_matrix(L)
# L = get_infectivity_matrix(L, vscales = 1./clusters.loc[clusters.index, 'area'].to_numpy())

L_sg = np.copy(L)

scales_nu_sg, Ss_nu_sg, Vs_nu_sg = fit_scales_nu(len(times), L_sg, Ss_real[0], Vs_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_sg, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nufit_safegraph' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nu_sg, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nu_sg, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nu_sg, columns=['scale'])

#--------------------------------------------------------------------------------
# uniform infectivity matrix
#--------------------------------------------------------------------------------
N = len(clusters)
L_unif = np.ones((N,N), dtype=np.float_)

scales_nu_unif, Ss_nu_unif, Vs_nu_unif = fit_scales_nu(len(times), L_unif, Ss_real[0], Vs_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_uniform' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_unif, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nufit_uniform' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nu_unif, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nu_unif, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nu_unif, columns=['scale'])


#--------------------------------------------------------------------------------
# optimal infectivity matrix
#--------------------------------------------------------------------------------

# read the optimal matrices and average them
N = len(clusters)
L = np.zeros((N,N), dtype=np.float_)
Nt = len(times)-1
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  # for i in range(len(times)-1):
  for i in range(Nt):
    t = times[i]
    mykey = Path('/fit') / 'betamat_sifit' / 'infectivity_matrices' / t.strftime(tfmt)
    mykey = str(mykey)
    L += store[mykey].to_numpy().astype('float64')
L /= Nt

L_opt = np.copy(L)

scales_nu_opt, Ss_nu_opt, Vs_nu_opt = fit_scales_nu(len(times), L_opt, Ss_real[0], Vs_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_optimal' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_opt, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nufit_optimal' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nu_opt, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nu_opt, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nu_opt, columns=['scale'])


#--------------------------------------------------------------------------------
# distance truncated infectivity matrix
#--------------------------------------------------------------------------------

phi = np.int_(distances < dc*1.0e3)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_sg = store[mykey].to_numpy().astype('float64')

L_trunc = L_sg * phi

scales_nu_trunc, Ss_nu_trunc, Vs_nu_trunc = fit_scales_nu(len(times), L_trunc, Ss_real[0], Vs_real, gamma, 1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_distance_trunc_{:.0f}km'.format(dc) / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_trunc, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nufit_distance_trunc_{:.0f}km'.format(dc) / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nu_trunc, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nu_trunc, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nu_trunc, columns=['scale'])

