## Imports and global variables

from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
import datetime
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, bracket, curve_fit, Bounds
import scipy.stats as sst

sys.path.append(str(Path('..') / 'code'))

from functions import func_sir_dX, sir_SI_to_X, sir_X_to_SI, func_sir_dV, guess_scale, fsigmoid, fsigmoid_jac, framp2, framp2_jac, get_dTs

resdir = Path('../results/')
if not resdir.is_dir():
  raise ValueError('No results directory!')

resfile = resdir / 'safegraph_analysis.hdf5'
complevel=7
complib='zlib'
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  print(f"File {resfile.stem} has {len(store.keys())} entries.")

## Global variables and other quantities

gamma = 1/10.
ti = '2020-03-01'
# tf = '2020-09-01'
tf = '2021-02-16'

tfmt = '%Y-%m-%d'
ti = datetime.datetime.strptime(ti, tfmt)
tf = datetime.datetime.strptime(tf, tfmt)

### Load clusters to get population

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

### Load CSSEGI data

path = '/clustering/cssegi'
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  df_cssegi = store[path]

times = df_cssegi.index
idx = (times >= ti) & (times <= tf)
df_cssegi.drop(index=times[~idx], inplace=True)
times = df_cssegi.index.to_pydatetime().tolist()
df_cssegi

omega_real = df_cssegi.to_numpy().astype('float64')
domega_real = np.diff(omega_real, axis=0)
domega_real = np.concatenate([omega_real[0].reshape(1,-1), domega_real], axis=0)

# compute the real epidemic sizes per community through time
Ts_real = np.einsum('ta,a->ta', omega_real, population_inv)
Ss_real = 1. - Ts_real
Ss_real[~(Ss_real > 0.)] = np.finfo(np.float64).resolution
dTs_real = np.einsum('ta,a->ta', domega_real, population_inv)
T_tot_real = np.einsum('ta,a->t', Ts_real, population) / np.sum(population)
S_tot_real = np.einsum('ta,a->t', Ss_real, population) / np.sum(population)
dT_tot_real = np.einsum('ta,a->t', dTs_real, population) / np.sum(population)

X = clusters.index.to_numpy()
Y = np.einsum('ta,a->ta', 1-Ss_real, population)[0]
idx = Y > 0

# Define the V-vector
Vs_real = -np.log(Ss_real/Ss_real[0])

# dc = 200
dc = 400
# dc = 800
with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  distances = store['/clustering/distances'].to_numpy().astype('float64')
#================================================================================
## Integrate the dynamics based on the (S,I) variables
#================================================================================

def func_J_T(theta, L, gamma, times, Ss_real, population, method_solver='DOP853', dt=1., SIscale=False):

  Nt = len(times)
  t_eval = np.arange(Nt, dtype=np.float_)

  scales = framp2(t_eval[:-1], *theta)
  Xs = np.zeros((Nt, 2*N), dtype=np.float_)

  ptot_sq = np.sum(population**2)

  # initialization
  Xs[0] = sir_SI_to_X(Ss_real[0], 1.-Ss_real[0])

  # integration loop
  for i in range(1, Nt):
    Xs[i] = Xs[i-1] + dt*func_sir_dX(0., Xs[i-1], scales[i-1]*L, gamma)


  SIs = np.array([sir_X_to_SI(x,N) for x in Xs])
  Ss = SIs[:,0]
  Is = SIs[:,1]

  Ts = 1.-Ss

  U = np.einsum('ta,a->ta',(Ss-Ss_real), population)[1:]

  if SIscale:
    return np.einsum('ta,ta', U, U) / (len(times)*N), Ss, Is, scales
#         return np.einsum('ta,ta', U, U) / (len(times)*ptot_sq), Ss, Is, scales
  else:
    return np.einsum('ta,ta', U, U) / (len(times)*N)
#         return np.einsum('ta,ta', U, U) / (len(times)*ptot_sq)

def func_dJ_T(theta, L, gamma, times, Ss_real, population, method_solver='DOP853', dt=1.):

  Nt = len(times)
  t_eval = np.arange(Nt, dtype=np.float_)

  scales = framp2(t_eval[:-1], *theta)
  Xs = np.zeros((Nt, 2*N), dtype=np.float_)

  ptot_sq = np.sum(population**2)

  # initialization
  Xs[0] = sir_SI_to_X(Ss_real[0], 1-Ss_real[0])

  # integration loop
  for i in range(1, Nt):
    Xs[i] = Xs[i-1] + dt*func_sir_dX(0., Xs[i-1], scales[i-1]*L, gamma)

  SIs = np.array([sir_X_to_SI(x,N) for x in Xs])
  Ss = SIs[:,0]
  Is = SIs[:,1]

  U = np.einsum('ta,a->ta',(Ss_real-Ss), population)[1:]
  J = np.einsum('ta,ta', U, U) / (len(times)*N)

  # compute gradient
#     V = -2*dt*np.einsum('a,ta,ab,tb,ta->t', population**2, Ss[:-1], L, Is[:-1], Ts_real[1:] - Ts[1:]) / (len(times)*ptot_sq)
  V = -2*dt*np.einsum('a,ta,ab,tb,ta->t', population**2, Ss[:-1], L, Is[:-1], Ss[1:] - Ss_real[1:]) / (len(times)*N)
  W = framp2_jac(t_eval[:-1], *theta)

  return np.einsum('t,ti->i', V, W)

def fitsir_ramp_T(times, L, population, Ss_r, g, theta, dt=1., method_solver='DOP853'):
  # define function to minimize
  func_min = lambda c: func_J_T(np.array([theta[0], theta[1], c, theta[3]]), \
      L, g, times, Ss_r, population, method_solver='DOP853', dt=1., SIscale=False)

  xa, xb, xc, fa, fb, fc, funcalls = bracket(func_min, xa=theta[2], xb=2*theta[2])
  res = minimize_scalar(func_min, bracket=(xa,xb,xc), method='brent')
  theta[2] = res.x

  J, Ss, Is, scales = func_J_T(theta, L, g, times, Ss_r, population, SIscale=True)
  Vs = -np.log(Ss/Ss[0])

  return Ss, Is, Vs, scales, J

#--------------------------------------------------------------------------------
# SafeGraph infectivity matrix
#--------------------------------------------------------------------------------

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_sg = store[mykey].to_numpy().astype('float64')

  path = Path('/fit') / 'scales' / 'sifit_safegraph' / 'result'
  mykey = str(path / 'susceptible')
  Ss_sifit_sg = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'infected')
  Is_sifit_sg = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'nu')
  Vs_sifit_sg = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'scales')
  scales_sifit_sg = store[mykey].to_numpy().astype('float64').ravel()

Xfit = np.array([(t - times[0]).days for t in times[:-1]])
Y = scales_sifit_sg.copy()
theta_sig = [np.mean(Y),20.,0.,1.]

res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')

theta_sig = res[0]
Yfit = fsigmoid(Xfit, *theta_sig)

if (theta_sig[0] < 0.):
  a,b,c,d = theta_sig
  theta_sig[0] = -a
  theta_sig[2] = c+a
  theta_sig[3] = -d

theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
theta = res[0]

# Yfit = fsigmoid(Xfit, *theta_sig)
# plt.plot(Xfit, Y)
# plt.plot(Xfit, Yfit)
# Zfit = framp2(Xfit, *theta)
# plt.plot(Xfit, Zfit)

print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing siramp_sg")
Ss_siramp_sg, Is_siramp_sg, Vs_siramp_sg, scales_siramp_sg, J_siramp_sg = fitsir_ramp_T(times, L_sg, population, Ss_real, gamma, theta, dt=1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'siramp_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_sg, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'siramp_safegraph' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_siramp_sg, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_siramp_sg, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_siramp_sg, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_siramp_sg, columns=['scale'])

#--------------------------------------------------------------------------------
# uniform infectivity matrix
#--------------------------------------------------------------------------------

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_uniform' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_unif = store[mykey].to_numpy().astype('float64')

  path = Path('/fit') / 'scales' / 'sifit_uniform' / 'result'
  mykey = str(path / 'susceptible')
  Ss_sifit_unif = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'infected')
  Is_sifit_unif = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'nu')
  Vs_sifit_unif = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'scales')
  scales_sifit_unif = store[mykey].to_numpy().astype('float64').ravel()

Xfit = np.array([(t - times[0]).days for t in times[:-1]])
Y = scales_sifit_unif.copy()
theta_sig = [np.mean(Y),20.,0.,1.]

res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')

theta_sig = res[0]
Yfit = fsigmoid(Xfit, *theta_sig)

if (theta_sig[0] < 0.):
  a,b,c,d = theta_sig
  theta_sig[0] = -a
  theta_sig[2] = c+a
  theta_sig[3] = -d

theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
theta = res[0]

# Yfit = fsigmoid(Xfit, *theta_sig)
# plt.plot(Xfit, Y)
# plt.plot(Xfit, Yfit)
# Zfit = framp2(Xfit, *theta)
# plt.plot(Xfit, Zfit)

print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing siramp_unif")
Ss_siramp_unif, Is_siramp_unif, Vs_siramp_unif, scales_siramp_unif, J_siramp_unif = fitsir_ramp_T(times, L_unif, population, Ss_real, gamma, theta, dt=1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'siramp_uniform' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_unif, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'siramp_uniform' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_siramp_unif, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_siramp_unif, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_siramp_unif, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_siramp_unif, columns=['scale'])

# #--------------------------------------------------------------------------------
# # Optimal infectivity matrix
# #--------------------------------------------------------------------------------
#
# with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
#   path = Path('/fit') / 'scales' / 'sifit_optimal' / 'infectivity_matrices'
#   mykey = str(path / times[0].strftime(tfmt))
#   L_opt = store[mykey].to_numpy().astype('float64')
#
#   path = Path('/fit') / 'scales' / 'sifit_optimal' / 'result'
#   mykey = str(path / 'susceptible')
#   Ss_sifit_opt = store[mykey].to_numpy().astype('float64')
#
#   mykey = str(path / 'infected')
#   Is_sifit_opt = store[mykey].to_numpy().astype('float64')
#
#   mykey = str(path / 'nu')
#   Vs_sifit_opt = store[mykey].to_numpy().astype('float64')
#
#   mykey = str(path / 'scales')
#   scales_sifit_opt = store[mykey].to_numpy().astype('float64').ravel()
#
# Xfit = np.array([(t - times[0]).days for t in times[:-1]])
# Y = scales_sifit_opt.copy()
# theta_sig = [np.mean(Y), 10.,0.,1.]
#
# res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')
#
# theta_sig = res[0]
# Yfit = fsigmoid(Xfit, *theta_sig)
#
# if (theta_sig[0] < 0.):
#   a,b,c,d = theta_sig
#   theta_sig[0] = -a
#   theta_sig[2] = c+a
#   theta_sig[3] = -d
#
# print(theta_sig)
# # theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
# theta = [-theta_sig[3]/100., theta_sig[1], +theta_sig[0]*2., theta_sig[2]]
# res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
# theta = res[0]
#
# # Yfit = fsigmoid(Xfit, *theta_sig)
# # plt.plot(Xfit, Y)
# # plt.plot(Xfit, Yfit)
# # Zfit = framp2(Xfit, *theta)
# # plt.plot(Xfit, Zfit)
#
# print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing siramp_opt")
# Ss_siramp_opt, Is_siramp_opt, Vs_siramp_opt, scales_siramp_opt, J_siramp_opt = fitsir_ramp_T(times, L_opt, population, Ss_real, gamma, theta, dt=1.)
#
# with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
#   path = Path('/fit') / 'scales' / 'siramp_optimal' / 'infectivity_matrices'
#   mykey = str(path / times[0].strftime(tfmt))
#   store[mykey] = pd.DataFrame(index=clusters.index, data=L_opt, columns=clusters.index)
#
#   path = Path('/fit') / 'scales' / 'siramp_optimal' / 'result'
#   mykey = str(path / 'susceptible')
#   store[mykey] = pd.DataFrame(index=times, data=Ss_siramp_opt, columns=clusters.index)
#
#   mykey = str(path / 'infected')
#   store[mykey] = pd.DataFrame(index=times, data=Is_siramp_opt, columns=clusters.index)
#
#   mykey = str(path / 'nu')
#   store[mykey] = pd.DataFrame(index=times, data=Vs_siramp_opt, columns=clusters.index)
#
#   mykey = str(path / 'scales')
#   store[mykey] = pd.DataFrame(index=times[:-1], data=scales_siramp_opt, columns=['scale'])

#--------------------------------------------------------------------------------
# Distance truncated infectivity matrix
#--------------------------------------------------------------------------------

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'sifit_distance_trunc_{:.0f}km'.format(dc) / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_trunc = store[mykey].to_numpy().astype('float64')

  path = Path('/fit') / 'scales' / 'sifit_distance_trunc_{:.0f}km'.format(dc) / 'result'
  mykey = str(path / 'susceptible')
  Ss_sifit_trunc = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'infected')
  Is_sifit_trunc = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'nu')
  Vs_sifit_trunc = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'scales')
  scales_sifit_trunc = store[mykey].to_numpy().astype('float64').ravel()

Xfit = np.array([(t - times[0]).days for t in times[:-1]])
Y = scales_sifit_trunc.copy()
theta_sig = [np.mean(Y),20.,0.,1.]

res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')

theta_sig = res[0]
Yfit = fsigmoid(Xfit, *theta_sig)

if (theta_sig[0] < 0.):
  a,b,c,d = theta_sig
  theta_sig[0] = -a
  theta_sig[2] = c+a
  theta_sig[3] = -d

theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
theta = res[0]

# Yfit = fsigmoid(Xfit, *theta_sig)
# plt.plot(Xfit, Y)
# plt.plot(Xfit, Yfit)
# Zfit = framp2(Xfit, *theta)
# plt.plot(Xfit, Zfit)

print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing siramp_trunc")
Ss_siramp_trunc, Is_siramp_trunc, Vs_siramp_trunc, scales_siramp_trunc, J_siramp_trunc = fitsir_ramp_T(times, L_trunc, population, Ss_real, gamma, theta, dt=1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'siramp_distance_trunc' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_trunc, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'siramp_distance_trunc' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_siramp_trunc, columns=clusters.index)

  mykey = str(path / 'infected')
  store[mykey] = pd.DataFrame(index=times, data=Is_siramp_trunc, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_siramp_trunc, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_siramp_trunc, columns=['scale'])

#================================================================================
## Integrate the dynamics based on the nu variables
#================================================================================

def func_J_nu(theta, L, gamma, times, Ss_real, method_solver='DOP853', dt=1., Vscale=False):

  Nt = len(times)
  N = Ss_real.shape[1]
  t_eval = np.arange(Nt, dtype=np.float_)

  scales = framp2(t_eval[:-1], *theta)
  Vs = np.zeros((Nt, N), dtype=np.float_)

  Vs_real = -np.log(Ss_real/Ss_real[0])

  # integration loop
  for i in range(1, Nt):
    Vs[i] = Vs[i-1] + dt*func_sir_dV(0., Vs[i-1], scales[i-1]*L, gamma, Ss_real[0])
#         Vs[i] = Vs[i-1] + dt*scales[i-1]*np.einsum('ab,b', L, 1. - Ss_real[0]*np.exp(-Vs[i-1])) - gamma * dt * Vs[i-1]

  U = (Vs - Vs_real)[1:]

  J = np.einsum('ta,ta', U, U)

  if Vscale:
    return J, Vs, scales
  else:
    return J


def func_dJ_nu(theta, L, gamma, times, Ss_real, method_solver='DOP853', dt=1.):

  Nt = len(times)
  t_eval = np.arange(Nt, dtype=np.float_)

  scales = framp2(t_eval[:-1], *theta)
  Vs = np.zeros((Nt, N), dtype=np.float_)

  Vs_real = -np.log(Ss_real/Ss_real[0])

  # integration loop
  for i in range(1, Nt):
    Vs[i] = Vs[i-1] + dt*func_sir_dV(0., Vs[i-1], scales[i-1]*L, gamma, Ss_real[0])

  U = (Vs - Vs_real)[1:]
  J = np.einsum('ta,ta', U, U)

  # compute gradient
  V = 2*dt*np.einsum('ab,tb,ta->t', L, 1.-Ss[:-1], Vs[1:] - Vs_real[1:])
#     V = 2*scales
#     V = 2*np.einsum('ta,ta->t', Acoef, U)
  W = framp2_jac(t_eval[:-1], *theta)

  return np.einsum('t,ti->i', V, W)

def fitsir_ramp_nu(times, L, Ss_r, g, theta, dt=1., method_solver='DOP853'):

  # define function to minimize
  func_min = lambda c: func_J_nu(np.array([theta[0], theta[1], c, theta[3]]), \
      L, g, times, Ss_r, method_solver='DOP853', dt=1., Vscale=False)

  xa, xb, xc, fa, fb, fc, funcalls = bracket(func_min, xa=theta[2], xb=2*theta[2])
  res = minimize_scalar(func_min, bracket=(xa,xb,xc), method='brent')
  theta[2] = res.x

  J, Vs, scales = func_J_nu(theta, L, g, times, Ss_r, Vscale=True)

  Ss = Ss_r[0]*np.exp(-Vs)

  return Ss, Vs, scales, J

#--------------------------------------------------------------------------------
# SafeGraph infectivity matrix
#--------------------------------------------------------------------------------

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_sg = store[mykey].to_numpy().astype('float64')

  path = Path('/fit') / 'scales' / 'nufit_safegraph' / 'result'
  mykey = str(path / 'susceptible')
  Ss_nufit_sg = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'nu')
  Vs_nufit_sg = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'scales')
  scales_nufit_sg = store[mykey].to_numpy().astype('float64').ravel()

Xfit = np.array([(t - times[0]).days for t in times[:-1]])
Y = scales_nufit_sg.copy()
theta_sig = [np.mean(Y), 20.,0.,1.]

res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')

theta_sig = res[0]
Yfit = fsigmoid(Xfit, *theta_sig)

if (theta_sig[0] < 0.):
  a,b,c,d = theta_sig
  theta_sig[0] = -a
  theta_sig[2] = c+a
  theta_sig[3] = -d

theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
theta = res[0]

# Yfit = fsigmoid(Xfit, *theta_sig)
# plt.plot(Xfit, Y)
# plt.plot(Xfit, Yfit)
# Zfit = framp2(Xfit, *theta)
# plt.plot(Xfit, Zfit)

print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing nuramp_sg")
Ss_nuramp_sg, Vs_nuramp_sg, scales_nuramp_sg, J_nuramp_sg = fitsir_ramp_nu(times, L_sg, Ss_real, gamma, theta, dt=1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nuramp_safegraph' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_sg, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nuramp_safegraph' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nuramp_sg, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nuramp_sg, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nuramp_sg, columns=['scale'])

#--------------------------------------------------------------------------------
# Uniform infectivity matrix
#--------------------------------------------------------------------------------

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_uniform' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_unif = store[mykey].to_numpy().astype('float64')

  path = Path('/fit') / 'scales' / 'nufit_uniform' / 'result'
  mykey = str(path / 'susceptible')
  Ss_nufit_unif = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'nu')
  Vs_nufit_unif = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'scales')
  scales_nufit_unif = store[mykey].to_numpy().astype('float64').ravel()

Xfit = np.array([(t - times[0]).days for t in times[:-1]])
Y = scales_nufit_unif.copy()
theta_sig = [np.mean(Y), 20.,0.,1.]

res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')

theta_sig = res[0]
Yfit = fsigmoid(Xfit, *theta_sig)

if (theta_sig[0] < 0.):
  a,b,c,d = theta_sig
  theta_sig[0] = -a
  theta_sig[2] = c+a
  theta_sig[3] = -d

theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
theta = res[0]

# Yfit = fsigmoid(Xfit, *theta_sig)
# plt.plot(Xfit, Y)
# plt.plot(Xfit, Yfit)
# Zfit = framp2(Xfit, *theta)
# plt.plot(Xfit, Zfit)

print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing nuramp_unif")
Ss_nuramp_unif, Vs_nuramp_unif, scales_nuramp_unif, J_nuramp_unif = fitsir_ramp_nu(times, L_unif, Ss_real, gamma, theta, dt=1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nuramp_uniform' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_unif, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nuramp_uniform' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nuramp_unif, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nuramp_unif, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nuramp_unif, columns=['scale'])

#--------------------------------------------------------------------------------
# Distance truncated infectivity matrix
#--------------------------------------------------------------------------------

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nufit_distance_trunc_{:.0f}km'.format(dc) / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  L_trunc = store[mykey].to_numpy().astype('float64')

  path = Path('/fit') / 'scales' / 'nufit_distance_trunc_{:.0f}km'.format(dc) / 'result'
  mykey = str(path / 'susceptible')
  Ss_nufit_trunc = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'nu')
  Vs_nufit_trunc = store[mykey].to_numpy().astype('float64')

  mykey = str(path / 'scales')
  scales_nufit_trunc = store[mykey].to_numpy().astype('float64').ravel()

Xfit = np.array([(t - times[0]).days for t in times[:-1]])
Y = scales_nufit_trunc.copy()
theta_sig = [np.mean(Y), 20.,0.,1.]

res = curve_fit(fsigmoid, Xfit, Y, p0=theta_sig, jac=fsigmoid_jac, maxfev=1000, method='lm')

theta_sig = res[0]
Yfit = fsigmoid(Xfit, *theta_sig)

if (theta_sig[0] < 0.):
  a,b,c,d = theta_sig
  theta_sig[0] = -a
  theta_sig[2] = c+a
  theta_sig[3] = -d

theta = [-theta_sig[3], theta_sig[1], +theta_sig[0]/2., theta_sig[2]]
res = curve_fit(framp2, Xfit, Y, p0=theta, jac=framp2_jac, maxfev=1000, method='lm')
theta = res[0]

# Yfit = fsigmoid(Xfit, *theta_sig)
# plt.plot(Xfit, Y)
# plt.plot(Xfit, Yfit)
# Zfit = framp2(Xfit, *theta)
# plt.plot(Xfit, Zfit)

print((datetime.datetime.now() - datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S"), "minimizing nuramp_trunc")
Ss_nuramp_trunc, Vs_nuramp_trunc, scales_nuramp_trunc, J_nuramp_trunc = fitsir_ramp_nu(times, L_trunc, Ss_real, gamma, theta, dt=1.)

with pd.HDFStore(resfile, complevel=complevel, complib=complib) as store:
  path = Path('/fit') / 'scales' / 'nuramp_distance_trunc' / 'infectivity_matrices'
  mykey = str(path / times[0].strftime(tfmt))
  store[mykey] = pd.DataFrame(index=clusters.index, data=L_trunc, columns=clusters.index)

  path = Path('/fit') / 'scales' / 'nuramp_distance_trunc' / 'result'
  mykey = str(path / 'susceptible')
  store[mykey] = pd.DataFrame(index=times, data=Ss_nuramp_trunc, columns=clusters.index)

  mykey = str(path / 'nu')
  store[mykey] = pd.DataFrame(index=times, data=Vs_nuramp_trunc, columns=clusters.index)

  mykey = str(path / 'scales')
  store[mykey] = pd.DataFrame(index=times[:-1], data=scales_nuramp_trunc, columns=['scale'])


