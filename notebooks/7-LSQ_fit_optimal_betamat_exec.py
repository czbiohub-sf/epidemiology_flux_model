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

from functions import func_sir_dX, sir_SI_to_X, sir_X_to_SI, func_sir_dV, guess_scale

resdir = Path('../results/')
if not resdir.is_dir():
  raise ValueError('No results directory!')

resfile = resdir / 'safegraph_analysis.hdf5'
complevel=7
complib='zlib'
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  print(f"File {resfile.stem} has {len(store.keys())} entries.")

figdir = Path('../figures') / '6-LSQ_fit_optimal_betamat'
if not figdir.is_dir():
  figdir.mkdir(parents=True, exist_ok=True)

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

X = clusters.index.to_numpy()
Y = np.einsum('ta,a->ta', Ts_real, population)[0]
idx = Y > 0
plt.plot(X[idx],Y[idx], 'bo')

Ss_real[~(Ss_real > 0.)] = 1.0e-16
Vs_real = -np.log(Ss_real/Ss_real[0])

#================================================================================
## Integrate the dynamics based on the (S,I) variables
#================================================================================
def J_T(X, S0, I0, S1_real, population, dt=1.):
  '''
    Score function
  '''

  n = len(S1_real)
  B = np.reshape(X,(n,n), order='C')

  # method 1
  S1 = (1. - dt*np.einsum('ab,b', B, I0))*S0
  F = (S1 - S1_real)*population

  return np.einsum('a,a', F, F)


def dJ_T(X, S0, I0, S1_real, population, dt=1.):
  '''
    Gradient of score function
  '''
  n = len(S1_real)
  B = np.reshape(X,(n,n), order='C')

  # method 1
  S1 = (1. - dt*np.einsum('ab,b', B, I0))*S0
  F = (S1 - S1_real)*population

  grad = -2*dt*np.einsum('a,b->ab', S0*F*population, I0)

  return np.ravel(grad, order='C')

dt = 1.
method_solver = 'DOP853'
method_minimizer = 'L-BFGS-B'
bounds = Bounds(np.zeros(N**2), np.ones(N**2)*(+np.inf))

Ss = np.zeros((len(times), N), dtype=np.float_)
Is = np.zeros((len(times), N), dtype=np.float_)
Vs = np.zeros((len(times), N), dtype=np.float_)

# initialization
Is[0] = Ts_real[0]
Ss[0] = 1. - Is[0]
Xi = sir_SI_to_X(Ss[0], Is[0])
X = Xi.copy()

B = np.eye(N)
# B = L_SG.to_numpy()
ptot = np.sum(population)

for i in range(1, len(times)):
  print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "{:d} / {:d}".format(i, len(times)), np.all(np.isfinite(X)))

  # minimize the error at time t
  res = minimize(J_T, np.ravel(B, order='C'), args=(Ss[i-1], Is[i-1], Ss_real[i], population, dt), method=method_minimizer, jac=dJ_T, bounds=bounds)

  B = np.reshape(res.x, (N,N), order='C')

  # ODE solver : integrate the dynamics with the given infectivity matrix
  X = solve_ivp(func_sir_dX, y0=X, t_span=(0,dt), t_eval=[0, dt], \
      args=(B, gamma), method=method_solver).y[:,-1]

  Ss[i], Is[i] = sir_X_to_SI(X, N)
  Vs[i] = -np.log(Ss[i]/Ss[0])

  # save matrix
  with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
    path = Path('/fit') / 'betamat_sifit' / 'infectivity_matrices'
    mykey = str(path / times[i-1].strftime(tfmt))
    store[mykey] = pd.DataFrame(index=clusters.index, data=B, columns=clusters.index)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'betamat_sifit' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs, columns=clusters.index)

#================================================================================
## Integrate the dynamics based on the nu variables
#================================================================================
def J_nu(X, V0, V1_real, si_real, g, dt=1.):
  '''
  Score function
  '''
  n = len(V1_real)
  B = np.reshape(X,(n,n), order='C')

  Y0 = 1. - si_real*np.exp(-V0)
  V1 = (1. - gamma*dt)*V0 + dt*np.einsum('ab,b', B, Y0)
  F = (V1 - V1_real)

  return np.einsum('a,a', F, F)

def dJ_nu(X, V0, V1_real, si_real, g, dt=1.):
  '''
  Gradient of score function
  '''
  n = len(V1_real)
  B = np.reshape(X,(n,n), order='C')

  Y0 = 1. - si_real*np.exp(-V0)
  V1 = (1. - gamma*dt)*V0 + dt*np.einsum('ab,b', B, Y0)
  F = (V1 - V1_real)

  grad = 2*dt*np.einsum('a,b->ab', F, Y0)

  return np.ravel(grad, order='C')


dt = 1.
method_solver = 'DOP853'
method_minimizer = 'L-BFGS-B'
bounds = Bounds(np.zeros(N**2), np.ones(N**2)*(+np.inf))

Vs = np.zeros((len(times), N), dtype=np.float_)

# initialization
Vs[0] = Vs_real[0]

B = np.eye(N)

for i in range(1, len(times)):
  print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "{:d} / {:d}".format(i, len(times)), np.all(np.isfinite(X)))

  # minimize the error at time t
  res = minimize(J_nu, np.ravel(B, order='C'), args=(Vs[i-1], Vs_real[i], Ss_real[0], gamma, dt), \
      method=method_minimizer, jac=dJ_nu, bounds=bounds)

  B = np.reshape(res.x, (N,N), order='C')

  # ODE solver : integrate the dynamics with the given infectivity matrix
  Vs[i] = solve_ivp(func_sir_dV, y0=Vs[i-1], t_span=(0,dt), t_eval=[0, dt], \
      args=(B, gamma, Ss_real[0]), method=method_solver).y[:,-1]
  #     Vs[i] = Vs[i-1] + func_sir_dV((i-1)*dt, Vs[i-1], B, gamma, Ss_real[0])

  # save matrix
  with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
    path = Path('/fit') / 'betamat_nufit' / 'infectivity_matrices'
    mykey = str(path / times[i-1].strftime(tfmt))
    store[mykey] = pd.DataFrame(index=clusters.index, data=B, columns=clusters.index)

Ss = np.einsum('ta,a->ta', np.exp(-Vs), Ss_real[0])

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'betamat_nufit' / 'result'

  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs, columns=clusters.index)
