#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Dec 2020

@author: Guillaume Le Treut

"""
#==============================================================================
# libraries
#==============================================================================
import os
import copy
import re
import pickle as pkl
import numpy as np
try:
  import cupy as cp
except ImportError:
  cp = None
import pandas as pd
import datetime
from pathlib import Path
import shutil
import scipy.integrate
import scipy.stats as sst
import scipy.special as ssp
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.cm as cm
import matplotlib.colors as mco
import matplotlib.patches as mpatches
import matplotlib.colors as mco
import matplotlib.ticker as ticker
from matplotlib import animation

import imageio

# #==============================================================================
# # helpers methods
# #==============================================================================
# def histogram(X,density=True):
#     valmax = np.max(X)
#     valmin = np.min(X)
#     iqrval = sst.iqr(X)
#     nbins_fd = (valmax-valmin)*np.float_(len(X))**(1./3)/(2.*iqrval)
#     if (nbins_fd < 1.0e4):
#         return np.histogram(X,bins='auto',density=density)
#     else:
#         print("Using 'sturges' method!")
#         return np.histogram(X,bins='sturges',density=density)
#
# def make_binning_edges(X, x0=None, x1=None, binw=None):
#     if x0 is None:
#         x0 = np.min(X)
#     if x1 is None:
#         x1 = np.max(X)
#
#     nx = len(X)
#     if binw is None:
#         nbins = np.ceil(np.sqrt(nx))
#         binw = (x1-x0)/nbins
#
#     nbins = float(x1-x0)/binw
#     nbins = int(np.ceil(nbins))
#     x1 = x0 + nbins*binw
#     edges = np.arange(nbins+1)*binw + x0
#
#     return edges
#
# def get_binned(X, Y, edges):
#     nbins = len(edges)-1
#     digitized = np.digitize(X,edges)
#     Y_subs = [None for n in range(nbins)]
#     for i in range(1, nbins+1):
#         Y_subs[i-1] = np.array(Y[digitized == i])
#
#     return Y_subs
#
# def get_array_module(a):
#   """
#   Return the module of an array a
#   """
#   if cp:
#     return cp.get_array_module(a)
#   else:
#     return np
#
# #==============================================================================
# # modelling
# #==============================================================================
# ########## localization methods ##########
# def entropy(data, nbins=None, x0=-1., x1=1.):
#     """
#     compute entropy of input data
#     """
#     s = np.std(data)
#     # hist, edges = histogram(data)
#     hist, edges = np.histogram(data, bins='doane', density=True)
#     # if nbins is None:
#     #     hist, edges = np.histogram(data, density=True)
#     # else:
#     #     bins = x0 + np.arange(nbins+1)*(x1-x0)/nbins
#     #     hist, edges = np.histogram(data, bins=bins, density=True)
#     dx = np.diff(edges)[0]
#     # print(f"dx: {dx}    nbins: {len(hist)}")
#
#     return -np.sum(ssp.xlogy(hist,hist))*dx - np.log(s)    # - \sum x log(x) (unit variance)
#
# ########## fitting methods ##########
# def fsigmoid(x, *params):
#     a, b, c, d = list(params)
#     return a / (1.0 + np.exp(d*(x-b))) + c
#
# def fsigmoid_jac(x, *params):
#     a, b, c, d = list(params)
#     grad = np.zeros((len(params), len(x)))
#     grad[0] = 1./(1.0 + np.exp(d*(x-b)))
#     grad[1] = d*np.exp(d*(x-b)) * a /(1.0 + np.exp(d*(x-b)))**2
#     grad[2] = np.ones(len(x))
#     grad[3] = -(x-b)*np.exp(d*(x-b)) * a /(1.0 + np.exp(d*(x-b)))**2
#     return grad.T
#
# def fsigmoid2(x, *params):
#     a, b, c = list(params)
#     return a / (1.0 + np.exp(c*(x-b)))
#
# def fsigmoid2_jac(x, *params):
#     a, b, d = list(params)
#     grad = np.zeros((len(params), len(x)))
#     grad[0] = 1./(1.0 + np.exp(d*(x-b)))
#     grad[1] = d*np.exp(d*(x-b)) * a /(1.0 + np.exp(d*(x-b)))**2
#     grad[2] = -(x-b)*np.exp(d*(x-b)) * a /(1.0 + np.exp(d*(x-b)))**2
#     return grad.T
#
# def gmm_list_to_array(params):
#   p = list(params)
#   return np.reshape(p, (-1,3))
#
# def gmm_array_to_list(p):
#   return np.ravel(p)
#
# def fgmm(x, *params):
#   """
#   Gaussian mixture model fitting function.
#   """
#   p = gmm_list_to_array(params)
#   mus, sigmas, cs = p.T
#   return np.sum([np.exp(-0.5*(x-mu)**2/sigma**2 + c) for mu, sigma, c in zip(mus, sigmas, cs)], axis=0)
#
# def fgmm_jac(x, *params):
#   """
#   Gaussian mixture model fitting function.
#   This gradient was checked with a code like:
#   ```
#     x = np.arange(nv)
#     errs = []
#     for i in range(nv):
#         func = lambda pp: func_fit(pp, x)[i]
#         func_jac = lambda pp: func_fit_jac(pp, x)[i]
#         err = check_grad(func, func_jac, p)
#         errs.append(err)
#         np.linalg.norm(errs)
#   ```
#   """
#   p = gmm_list_to_array(params)
#   ngauss = p.shape[0]
#   if p.shape[1] != 3:
#     raise ValueError("Implemented with 3 adjustable parameter per Gaussian distribution.")
#
#   mus, sigmas, cs = p.T
#
#   grad = np.array([ [  (x-mu)/sigma**2 * np.exp(-0.5*(x-mu)**2/sigma**2 + c), \
#                       (x-mu)**2/sigma**3 * np.exp(-0.5*(x-mu)**2/sigma**2 + c), \
#                       np.exp(-0.5*(x-mu)**2/sigma**2 + c) \
#                     ] for mu, sigma, c in zip(mus, sigmas, cs) \
#                   ])
#
#   grad = np.reshape(grad, (3*ngauss, len(x)))
#   return grad.T
#
# ########## Utils ##########
# def read_df(t, tfmt, store, path):
#   """
#   Read a matrix present in a `store` at a certain `path` with
#   the appropriate formatting of the date `t`.
#   """
#   key = Path(path) / t.strftime(tfmt)
#   df = store[str(key)]
#   return df
#
# def get_population(clusters):
#   return clusters['population'].to_numpy().astype('int64')
#
# def get_infectivity_matrix(F):
#   """
#   Return the infectivity matrix from the input flux matrix
#   """
#   N = F.shape[0]
#   if (F.shape[1] != N):
#     raise ValueError
#
#   pvec = F.diagonal()
#   pinv = np.zeros(N, dtype=np.float_)
#   idx = pvec > 0.
#   pinv[idx] = 1./pvec[idx]
#
#   B = np.zeros((N,N), dtype=np.float_)
#   B = F + F.T
#   np.fill_diagonal(B, pvec)
#   B = np.einsum('ij,j,i->ij', B, pinv, pinv)
#   return B
#
# def get_localization_matrix(F):
#   """
#   Return the localization matrix from the input flux matrix
#   """
#   N = F.shape[0]
#   if (F.shape[1] != N):
#     raise ValueError
#
#   pvec = F.diagonal()
#   pinv = np.zeros(N, dtype=np.float_)
#   idx = pvec > 0.
#   pinv[idx] = 1./pvec[idx]
#
#   L = np.zeros((N,N), dtype=np.float_)
#   L = F + F.T
#   np.fill_diagonal(L, pvec)
#   L = np.einsum('ij,i->ij', L, pinv)
#
#   # symmetrize it
#   L = 0.5*(L+L.T)
#
#   return L
#
# def get_localization_matrix_vscale(F,V):
#   """
#   Return the localization matrix from the input flux matrix
#   INPUT:
#     * F: Flux matrix
#     * V: scale to apply
#   """
#   N = F.shape[0]
#   if (F.shape[1] != N):
#     raise ValueError
#
#   pvec = F.diagonal()
#   pinv = np.zeros(N, dtype=np.float_)
#   idx = pvec > 0.
#   pinv[idx] = 1./pvec[idx]
#
#   L = np.zeros((N,N), dtype=np.float_)
#   L = np.einsum('ij,j->ij', F, V)
#   L += L.T
#   L = np.einsum('ij,i->ij', L, pinv)
#   np.fill_diagonal(L, V)
#
#   # symmetrize it
#   # L = 0.5*(L+L.T)
#
#   return L
#
# ########## Basis rotation ##########
# def rotation_mat(N, i, j, theta, xp=np):
#   """
#   Generate a rotation matrix of size NxN.
#   The (i,j)th generator is used.
#   Theta is the angle applied.
#   """
#
#   # test on validity of arguments
#   if not ( i < j):
#     raise ValueError("Arguments must be such that i<j!")
#
#   R = xp.eye(N, dtype=xp.float_)
#   R[i,i] = xp.cos(theta)
#   R[j,i] = -xp.sin(theta)
#   R[i,j] = xp.sin(theta)
#   R[j,j] = xp.cos(theta)
#
#   return R
#
# def vector_localize_rotations(v_, w_, itermax, idump=1, batch_size=2**7, macheps=np.finfo(float).resolution, seed=123, gpu_device=0):
#   """
#   This method is an algorithm for localizing the first eigenvector of the eigenbasis v, namely v[:,0].
#   At every step, the rotation plane is (0,j) where j is chosen so as to maximize the localization.
#   """
#   try:
#     import cupy as cp
#   except ImportError:
#     raise ValueError("Cupy needs to be available")
#
#   # set the seed
#   cp.random.seed(seed)
#
#   # use the given gpu device
#   cp.cuda.Device(gpu_device)
#
#   # copy vector
#   v = cp.asarray(v_)
#   N = v.shape[0]
#   if v.shape[1] != N:
#     raise ValueError("Input matrix v must be square.")
#
#   w = cp.asarray(w_)
#   if w.shape != (N,):
#     raise ValueError("Input w must be a N-vector.")
#
#   # check that the batch size is smaller than the size of the matrix
#   if batch_size > N:
#     batch_size = N
#     print("Decreasing batch size to {:d}".format(batch_size))
#
#   # set a cutoff on zero values
#   idx = cp.abs(v) < macheps
#   v[idx] = 0.
#
#   # make sure v is compatible with a positive matrix (Perron-Froebenius)
#   if cp.any(v[:,0] < 0.):
#     raise ValueError("v[:,0] should only have positive values!")
#
#   # localization function
#   # func_loc = lambda x: cp.linalg.norm(x, ord=np.inf)
#   # func_loc = lambda x: cp.linalg.norm(x, ord=4)**4
#
#   # initial condition
#   v0 = cp.copy(v)
#   L0 = cp.einsum('ia,a,ja->ij',v,w,v)
#   f0 = float(cp.linalg.norm(L0))
#
#   # place holders
#   psi_list = [v[:,0]]
#   data = []             # holder for (j, \theta, loc)
#
#   # loop
#   for it in range(itermax):
#     output=[]
#     output.append("iteration {:d} / {:d}".format(it+1, itermax))
#
#     # ensemble of indices with v[:,j] > 0
#     idx_1 = v > 0.
#
#     vfirst = cp.array([v[:,0]]*N).T
#     idx_2 = vfirst == 0.
#
#     # Intersection of positive v[i,j] and zero v[i,0]. Only js with empty intersection are kept.
#     # Nz is the cardinal of the intersection for every j
#     Nz = cp.einsum('ij,ij->j',idx_2.astype('int'), idx_1.astype('int'))
#     js = np.sort(np.ravel(np.argwhere(Nz.get() == 0.)))
#     if js[0] != 0:
#       raise ValueError("First value should be zero in js!")
#     js = js[1:]
#     if len(js) == 0:
#       raise ValueError("len(js) = 0!")
#
#     # compute angles
#     thetas = cp.ones((N,N), dtype=cp.float_)
#     thetas *= 99.9e99
#     thetas[idx_1] = cp.arctan(vfirst[idx_1]/v[idx_1])
#     thetas = cp.min(thetas, axis=0)
#     if (float(cp.min(thetas[js])) == 0.):
#       raise ValueError("theta must be > 0 here!")
#
#     # compute overlaps
#     # v2 = v**2
#     # overlaps = np.array([float(cp.einsum('i,i', v2[:,0], v2[:,j])) for j in js])
#
#     # compute rotated matrix for every angle and chose the one that increases the most the localization
#     # of the leading eigenvector
#     js_tp = []
#     loc_tp = []
#     for b in range(int(np.ceil(float(len(js))/batch_size))):
#       j0 = b*batch_size
#       j1 = j0+batch_size
#       Rs = cp.array([rotation_mat(N, 0, j, thetas[j], xp=cp) for j in js[j0:j1]])
#       vs = cp.einsum('ik,akj', v, Rs)
#       js_tp.append(js[j0:j1]),
#       loc_tp.append(cp.linalg.norm(vs[:,:,0], ord=np.inf, axis=1).get())
#       del Rs
#       del vs
#
#     js_tp = np.concatenate(js_tp)
#     loc_tp = np.concatenate(loc_tp)
#     if np.any(js != js_tp):
#       print("js", js)
#       print("js_tp", js_tp)
#       raise ValueError("js_tp must be equal to j!")
#
#     kmax = np.argmax(loc_tp)
#     loc_max = loc_tp[kmax]
#     j = js[kmax]
#     theta = float(thetas[j])
#
#     # overlap_max = np.max(overlaps)
#     # overlap_min = np.min(overlaps)
#     # overlap = overlaps[kmax]
#
#     R = rotation_mat(N, 0, j, theta, xp=cp)
#     v = cp.einsum('ik,kj', v, R)
#
#     # set a cutoff on zero values
#     idx = np.abs(v) < macheps
#     v[idx] = 0.
#
#     # dump quantities
#     loc = float(cp.linalg.norm(v[:,0], ord=np.inf))
#     L = cp.einsum('ia,a,ja->ij',v,w,v)
#     Ldist = float(cp.linalg.norm(L - L0)) / f0
#
#     psi_list.append(v[:,0])
#     data.append([j, theta, loc, Ldist])
#     output_fmt = ["j = {:d}", "theta = {:.6e}", "loc = {:.6e}", "|L-L0|/|L0| = {:.6e}"]
#
#     for fmt,el in zip(output_fmt, data[-1]):
#       output.append(fmt.format(el))
#
#     # output.append("overlaps (actual/max/min) = {:.6e}  {:.6e}  {:.6e}".format(overlap, overlap_max, overlap_min))
#
#     # print
#     if (it % idump ==0):
#       output_str = ["{:<25s}".format(el) for el in output]
#       output_str = "".join(output_str)# + "\n"
#       print(output_str)
#   #end loop
#
#   psi_list = cp.array(psi_list).get()
#   data = np.array(data)
#   v = v.get()
#
#   return v, psi_list, data
#
# def vector_fit_gmm(v, ngauss, npts_em=1000, nfit_em=10, seed=123, dolsq=True):
#   """
#   This method returns a fit of the input vector using a Gaussian Mixture Model.
#   The returned vector is normalized to the same norm as `v`.
#   The procedure is:
#   1) Perform an EM inference to get the initial guess
#   2) Refine with a least-square minimization.
#   """
#
#   ########## Initializations and test ##########
#   if np.any(v < 0.):
#     raise ValueError("v must be >=0 because it is assimilated to a pdf.")
#   nv = len(v)
#   x = np.arange(nv, dtype=np.float_)
#
#   ########## EM inference ##########
#   from scipy.stats import rv_histogram
#   from sklearn.mixture import GaussianMixture
#   # generate a fake data set using the input vector as an artificial pdf.
#   edges = np.arange(nv+1)
#   hist_dist = rv_histogram([v, edges])
#   data = hist_dist.rvs(size=npts_em)
#
#   # perform a GaussianMixture EM inference
#   gmm = GaussianMixture(n_components=ngauss, init_params='kmeans', n_init=nfit_em, random_state=seed)
#   res = gmm.fit(data.reshape(-1,1))
#
#   # save result of fit as a parameter vector to be passed to fgmm_fit
#   mus = res.means_.reshape(ngauss)
#   sigmas = np.sqrt(res.covariances_.reshape(ngauss))
#   scales = res.weights_.reshape(ngauss)
#   p0 = np.array([mus,sigmas,np.log(scales)]).T
#   p0 = gmm_array_to_list(p0)
#
#   ########## Least-square minimization ##########
#   p = p0.copy()
#   if dolsq:
#     from scipy.optimize import curve_fit
#     p, pcov = curve_fit(fgmm, x, v, p0=p0, jac=fgmm_jac, method='lm')
#
#     # the output returns std that can be negative, rectify that.
#     params = gmm_list_to_array(p)
#     params[:,1] = np.abs(params[:,1])
#     p = gmm_array_to_list(params)
#
#   ########## approximate vector ##########
#   # fix normalization to unit vector
#   f1 = np.linalg.norm(v)
#   v_approx = fgmm(x, *p)
#   f2 = np.linalg.norm(v_approx)
#   c0 = np.log(f1)-np.log(f2)
#   params = gmm_list_to_array(p)
#   params[:,2] = params[:,2] + c0
#
#   # sort Gaussian by their respective weights (descending)
#   idx = np.lexsort(params.T)[::-1]
#   params=params[idx]
#
#   p = gmm_array_to_list(params)
#
#   v_approx = fgmm(x, *p)
#   return v_approx, params
#
# def basis_update_v0(v, y, macheps=1.0e-10):
#   """
#   Perform the operation v[:,0] <- y and reconstruct an orthogonal basis using the QR decomposition (Gram-Schmidt)
#   INPUT:
#     * v is an orthogonal matrix
#     * y is a vector
#   """
#   N = v.shape[0]
#   # macheps = np.finfo(float).resolution
#   if v.shape[1] != N:
#     raise ValueError("v must be a square matrix!")
#   vTv = np.einsum('ki,kj', v, v)
#   err = np.linalg.norm(vTv-np.eye(N))/np.sqrt(N)
#   if not (err < macheps):
#     raise ValueError("v must be an orthogonal matrix! |vTv-I| = {:.6e}".format(err))
#
#   v_new = np.concatenate([y.reshape(-1,1),v[:,1:]], axis=1)
#   v_new, R = np.linalg.qr(v_new)
#
#   # adjust orientation of vectors of the new basis
#   overlaps = np.einsum('ij,ij->j', v, v_new)
#   signs = np.ones(N, dtype=np.int_)
#   signs[overlaps < 0.] = -1
#   v_new = np.einsum('ij,j->ij', v_new, signs)
#
#   return v_new
#
# ########## SIR integration ##########
# def sir_X_to_SI(X, N):
#   SI = X.reshape((2,N))
#   return SI[0],SI[1]
#
# def sir_SI_to_X(S,I):
#   return np.ravel(np.array([S,I]))
#
# def func_sir_dX(t, X, B, g):
#   """
#   X: S, I
#   B: localization matrix
#   g: inverse recovery time
#   """
#   N = B.shape[0]
#   S,I = sir_X_to_SI(X, N)
#
#   dS = -np.einsum('i,ij,j->i', S, B, I)
#   dI = -dS - g*I
#
#   # return np.array([dS, dI])
#   # return np.ravel(np.array([dS, dI]))
#   return sir_SI_to_X(dS,dI)
#
# def jac(X, B, g):
#   N = B.shape[0]
#   SI = X.reshape((2,N))
#   S = SI[0]
#   I = SI[1]
#
#   # derivative of f_S
#   A1 = -  np.diag(np.einsum('ij,j->i', B, I))
#   A2 = - np.einsum('ij,i->ij', B, S)
#   A = np.concatenate([A1, A2], axis=1)
#
#   # derivative of f_I
#   B1 = -A1
#   B2 = -A2 - g*np.eye(N)
#   B = np.concatenate([B1, B2], axis=1)
#
#   return np.concatenate([A,B], axis=0)
#
# def compute_sir_X(X, dt, B, g, method_solver, t_eval=None):
#   if t_eval is None:
#     t_eval = [0., dt]
#   sol = scipy.integrate.solve_ivp(func_dX, y0=X, t_span=(0, dt), t_eval=t_eval, \
#       jac=None, vectorized=True, args=(B, gamma), method=method_solver)
#
#   # break conditions
#   if not (sol.success):
#     raise ValueError("integration failed!")
#
#   Xnew = sol.y[:,1:].T
#   return Xnew
#
# def get_sir_omega_X(X, P):
#   """
#   Compute the total fraction of T=I+R individuals from local fractions and local populations
#   """
#   N = len(P)
#   return np.einsum('i,i', 1.-X.reshape(2, N)[0], P)/np.einsum('i->', P)
#
# def get_sir_omega_SI(S, I, P):
#   """
#   Compute the total fraction of I+R individuals from local fractions and local populations
#   """
#   return get_sir_omega_X(sir_SI_to_X(S, I), P)
#
# def compute_sir_X(X, dt, B, gamma, method_solver, t_eval=None):
#   """
#   Utility function to integrate the SIR dynamics by dt.
#   """
#   if t_eval is None:
#     t_eval = [0., dt]
#
#   sol = scipy.integrate.solve_ivp(func_sir_dX, y0=X, t_span=(0,dt), \
#                                   t_eval=t_eval, vectorized=True, args=(B, gamma), \
#                                   method=method_solver)
#
#   # break conditions
#   if not (sol.success):
#     raise ValueError("integration failed!")
#
#   Xnew = sol.y[:,-1:].T
#
#   return Xnew
#
# def get_epidemic_size(M, epsilon_i, gamma, itermax=1000, rtol_stop=1.0e-8):
#   """
#   Compute the epidemic size given an initial condition epsilon_i and a localization matrix M.
#   epsilon_i represents the fraction of infected individuals at time t=0, in each community.
#   """
#   N = M.shape[0]
#   if (M.shape[1] != N):
#     raise ValueError
#   if (len(epsilon_i) != N):
#     raise ValueError
#
#   Xnew = np.zeros(N, dtype=np.float_)
#   for iter in range(itermax):
#     X = Xnew.copy()
#     B = 1. - (1.-epsilon_i)*np.exp(-X)
#     Xnew = 1./gamma * np.einsum('ab,b', M, B)
#
#     rtol = np.linalg.norm(X-Xnew)/(np.linalg.norm(X)+np.linalg.norm(Xnew))*2
# #     print("rtol = {:.6e}".format(rtol))
#     if (rtol < rtol_stop):
#       break
#     if (iter == itermax -1):
#       #             raise ValueError("Algorithm didn't converge! rtol = {:.6e}".format(rtol))
#       print("Algorithm didn't converge! rtol = {:.6e}".format(rtol))
#
#   B = 1. - (1.-epsilon_i)*np.exp(-X)
#   Omega = np.sum(B) / N
#   return Omega
#
# def get_target_scale(M, Ii, gamma, target=0.1, rtol_stop=1.0e-8, itermax=100):
#   from scipy.optimize import root_scalar
#
#   # define function to zero
#   func_root = lambda x: get_epidemic_size(x*M, Ii, gamma, itermax=itermax, rtol_stop=rtol_stop) - target
#
#   # initial bracketing
#   xlo = 1.0e-5
#   flo = func_root(xlo)
#   if flo > 0.:
#     raise ValueError("Lower bound on scale not small enough!")
#   xhi = xlo
#   for k in range(10):
#     fhi = func_root(xhi)
#     if fhi > 0.:
#       break
#     else:
#       xhi *= 10
#   if fhi < 0.:
#     raise ValueError("Problem in bracketing!")
#
#   # root finding
#   sol = root_scalar(func_root, bracket=(xlo, xhi), method='brentq', options={'maxiter': 100})
#   return sol.root
#
# def integrate_sir(Xi, times, gamma, store, pathtoloc, tfmt='%Y-%m-%d', method_solver='DOP853', verbose=True):
#   """
#   Integrate the dynamics of the SIR starting from
#   the initial condition (`Xi`, `times[0]`).
#   The method assumes that in the `store` at the indicated `path`, there are entries
#   in the format %Y-%m-%d that described the localization matrices
#   for the times `times[:-1]`.
#
#   OUTPUT:
#     * Xs
#     * ts
#
#   For the output the dumping interval is 1 day.
#   """
#   # initializations
#   nt = len(times)
#   t = times[0]
#   X = Xi[:]
#   B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#   N = B.shape[0]
#
#   ts = [t]
#   Xs = [X]
#
#   for i in range(1, nt):
#     if verbose:
#       print(f'Integrating day {t}')
#     mykey = Path(pathtoloc) / t.strftime(tfmt)
#     mykey = str(mykey)
#     if mykey in store.keys():
#       B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#     elif verbose:
#       print("Localization matrix not updated!")
#     tnew = times[i]
#     dt = int((tnew - t).days)
#     t_range = np.arange(dt+1)
#     sol = scipy.integrate.solve_ivp(func_sir_dX, y0=X, t_span=(0,dt), \
#                                     t_eval=t_range, vectorized=True, args=(B, gamma), \
#                                     method=method_solver)
#
#     # break conditions
#     if not (sol.success):
#       raise ValueError("integration failed!")
#
#     Xnew = sol.y[:,-1]
#
#     # dump
#     Xs += [x for x in sol.y[:, 1:].T]
#     ts += [t + datetime.timedelta(days=int(x)) for x in t_range[1:]]
#
#     # update
#     t = tnew
#     X = Xnew
#
#   if verbose:
#     print("Integration complete")
#
#   SIs = np.array([sir_X_to_SI(x, N) for x in Xs])
#   Ss = SIs[:,0]
#   Is = SIs[:,1]
#
#   return ts, Ss, Is
#
# def fit_sir(times, T_real, gamma, population, store, pathtoloc, tfmt='%Y-%m-%d', method_solver='DOP853', verbose=True, \
#             b_scale=1):
#   """
#   Fit the dynamics of the SIR starting to real data contained in `pathtocssegi`.
#   The initial condition is taken from the real data.
#   The method assumes that in the `store` at the indicated `path`, there are entries
#   in the format %Y-%m-%d that described the localization matrices
#   for the times `times[:-1]`.
#   `populations` is the vector with the population per community.
#
#   OUTPUT:
#     * Xs
#     * ts
#     * scales
#
#   For the output the dumping interval is one day.
#   """
#
#   # initializations
#   nt = len(times)
#   t = times[0]
#   B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#   N = B.shape[0]
#   Y_real = np.einsum('ta,a->t', T_real, population) / np.sum(population)
#
#   X = np.zeros((2, N), dtype=np.float_)
#   I = T_real[0]
#   S = 1 - I
#   X = sir_SI_to_X(S, I)
#
#   y = get_sir_omega_X(X, population)
#
#   ts = [t]
#   Xs = [X.reshape(2,N)]
#   Ys = [y]
#   b_scales = []
#
#   blo = 0.
#   # print("nt = ", nt)
#
#   for i in range(1, nt):
#     if verbose:
#       print(f'Integrating day {t}')
#     mykey = Path(pathtoloc) / t.strftime(tfmt)
#     mykey = str(mykey)
#     if mykey in store.keys():
#       B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#     elif verbose:
#       print("Localization matrix not updated!")
#
#     tnew = times[i]
#     dt = int((tnew - t).days)
#     ypred = Y_real[i]
#
#     # root finding method
#     func_root = lambda b: get_sir_omega_X(compute_sir_X(X, dt, b*B, gamma, method_solver), \
#                                           population) - ypred
#
#     # initial bracketing
#     bhi = b_scale
#     fscale = 3.
#     for k in range(1,10):
#       f = func_root(bhi)
#       if f > 0:
#         break
#       else:
#         bhi *= fscale
#     if f < 0:
#       raise ValueError("Problem in bracketing!")
#
#     # find the root
#     sol = scipy.optimize.root_scalar(func_root, bracket=(blo, bhi), method='brentq', \
#                                       options={'maxiter': 100})
#     if not (sol.converged):
#       raise ValueError("root finding failed!")
#     b_scale = sol.root
#
#     # compute next state with optimal scale
#     t_eval = np.arange(dt+1)
#     Xnews = compute_sir_X(X, dt, b_scale*B, gamma, method_solver, t_eval=t_eval)
#     Xnew = Xnews[-1]
#     y = get_sir_omega_X(Xnew,population)
#     print(f"b = {b_scale}, y = {y}, ypred = {ypred}, y-ypred = {y-ypred}")
#
#     # dump
#     # data.append(Xnew.reshape(2,N))
#     Xs += [Xnew.reshape(2,N) for Xnew in Xnews]
#     ts += [t + datetime.timedelta(days=int(dt)) for dt in t_eval[1:]]
#     Ys.append(y)
#     b_scales.append(b_scale)
#
#     # update
#     t = tnew
#     X = Xnew
#
#   b_scales.append(None)  # B has ndays-1 entries
#   print("Fitting complete")
#
#   # prepare export of results
#   S = np.array([X[0] for X in Xs])
#   I = np.array([X[1] for X in Xs])
#   clusters = np.arange(N, dtype=np.uint)
#   df_S = pd.DataFrame(data=S, index=ts, columns=clusters)
#   df_I = pd.DataFrame(data=I, index=ts, columns=clusters)
#   df_fit = pd.DataFrame(data=np.array([b_scales, Ys]).T, index=times, columns=["localization_scale", "frac_infected_tot"])
#
#   return df_S, df_I, df_fit
#
#
# def fit_sir_test(times, T_real, gamma, population, store, pathtoloc, tfmt='%Y-%m-%d', method_solver='DOP853', verbose=True, \
#             b_scale=1, index=None):
#   """
#   Fit the dynamics of the SIR starting to real data contained in `pathtocssegi`.
#   The initial condition is taken from the real data.
#   The method assumes that in the `store` at the indicated `path`, there are entries
#   in the format %Y-%m-%d that described the localization matrices
#   for the times `times[:-1]`.
#   `populations` is the vector with the population per community.
#
#   OUTPUT:
#     * Xs
#     * ts
#     * scales
#
#   For the output the dumping interval is one day.
#   """
#
#   # initializations
#   nt = len(times)
#   t = times[0]
#   B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#   N = B.shape[0]
#   if index is None:
#     index = np.arange(N)
#   Y_real = np.einsum('ta,a->t', T_real[:,index], population[index]) / np.sum(population[index])
#
#   # X = np.zeros((2, N), dtype=np.float_)
#   I = T_real[0]
#   S = 1. - I
#   T = 1. - S
#   X = sir_SI_to_X(S, I)
#
#   y = np.einsum('a,a', T[index], population[index]) / np.sum(population[index])
#   # y = get_sir_omega_X(X, population)
#
#   ts = [t]
#   Xs = [X.reshape(2,N)]
#   Ys = [y]
#   b_scales = []
#
#   blo = 0.
#   # print("nt = ", nt)
#
#   for i in range(1, nt):
#     if verbose:
#       print(f'Integrating day {t}')
#     mykey = Path(pathtoloc) / t.strftime(tfmt)
#     mykey = str(mykey)
#     if mykey in store.keys():
#       B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#     elif verbose:
#       print("Localization matrix not updated!")
#
#     tnew = times[i]
#     dt = int((tnew - t).days)
#     ypred = Y_real[i]
#
#     # root finding method
#     def func_root(b):
#       X_new = compute_sir_X(X, dt, b*B, gamma, method_solver)
#       T = 1.- X_new.reshape(2, N)[0]
#       y = np.einsum('a,a', T[index], population[index]) / np.sum(population[index])
#       # y = get_sir_omega_X(X_new, population)
#       return y - ypred
#
#     # func_root = lambda b:  \
#     #     np.einsum('a,a', \
#     #     (1.- compute_sir_X(X, dt, b*B, gamma, method_solver).reshape(2, N)[0])[index], \
#     #     population[index]) / np.sum(population[index]) - ypred
#     # func_root = lambda b: get_sir_omega_X(compute_sir_X(X, dt, b*B, gamma, method_solver), \
#     #                                       population) - ypred
#
#     # initial bracketing
#     bhi = b_scale
#     fscale = 3.
#     for k in range(1,10):
#       f = func_root(bhi)
#       if f > 0:
#         break
#       else:
#         bhi *= fscale
#     if f < 0:
#       raise ValueError("Problem in bracketing!")
#
#     # find the root
#     sol = scipy.optimize.root_scalar(func_root, bracket=(blo, bhi), method='brentq', \
#                                       options={'maxiter': 100})
#     if not (sol.converged):
#       raise ValueError("root finding failed!")
#     b_scale = sol.root
#
#     # compute next state with optimal scale
#     t_eval = np.arange(dt+1)
#     Xnews = compute_sir_X(X, dt, b_scale*B, gamma, method_solver, t_eval=t_eval)
#     Xnew = Xnews[-1]
#     # y = get_sir_omega_X(Xnew,population)
#     T = 1.- Xnew.reshape(2, N)[0]
#     y = np.einsum('a,a', T[index], population[index]) / np.sum(population[index])
#     print(f"b = {b_scale}, y = {y}, ypred = {ypred}, y-ypred = {y-ypred}")
#
#     # dump
#     # data.append(Xnew.reshape(2,N))
#     Xs += [Xnew.reshape(2,N) for Xnew in Xnews]
#     ts += [t + datetime.timedelta(days=int(dt)) for dt in t_eval[1:]]
#     Ys.append(y)
#     b_scales.append(b_scale)
#
#     # update
#     t = tnew
#     X = Xnew
#
#   b_scales.append(None)  # B has ndays-1 entries
#   print("Fitting complete")
#
#   # prepare export of results
#   S = np.array([X[0] for X in Xs])
#   I = np.array([X[1] for X in Xs])
#   clusters = np.arange(N, dtype=np.uint)
#   df_S = pd.DataFrame(data=S, index=ts, columns=clusters)
#   df_I = pd.DataFrame(data=I, index=ts, columns=clusters)
#   df_fit = pd.DataFrame(data=np.array([b_scales, Ys]).T, index=times, columns=["localization_scale", "frac_infected_tot"])
#
#   return df_S, df_I, df_fit
#
# def fit_sir_variable_index(times, T_real, gamma, population, store, pathtoloc, tfmt='%Y-%m-%d', method_solver='DOP853', verbose=True, \
#             b_scale=1):
#   """
#   Fit the dynamics of the SIR starting to real data contained in `pathtocssegi`.
#   The initial condition is taken from the real data.
#   The method assumes that in the `store` at the indicated `path`, there are entries
#   in the format %Y-%m-%d that described the localization matrices
#   for the times `times[:-1]`.
#   `populations` is the vector with the population per community.
#
#   OUTPUT:
#     * Xs
#     * ts
#     * scales
#
#   For the output the dumping interval is one day.
#   """
#
#   # initializations
#   nt = len(times)
#   t = times[0]
#   B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#   N = B.shape[0]
#   # Y_real = np.einsum('ta,a->t', T_real[:,index], population[index]) / np.sum(population[index])
#   i = 0
#   index = T_real[i] > 0
#   ypred = np.einsum('a,a', T_real[i,index], population[index]) / np.sum(population[index])
#
#   # X = np.zeros((2, N), dtype=np.float_)
#   I = T_real[0]
#   S = 1. - I
#   T = 1. - S
#   X = sir_SI_to_X(S, I)
#
#   y = np.einsum('a,a', T[index], population[index]) / np.sum(population[index])
#   print("y = {:.4e}    y_real = {:.4e}".format(y, ypred))
#   # y = get_sir_omega_X(X, population)
#
#   ts = [t]
#   Xs = [X.reshape(2,N)]
#   Ys = [y]
#   b_scales = []
#
#   blo = 0.
#   # print("nt = ", nt)
#
#   for i in range(1, nt):
#     if verbose:
#       print(f'Integrating day {t}')
#     mykey = Path(pathtoloc) / t.strftime(tfmt)
#     mykey = str(mykey)
#     if mykey in store.keys():
#       B = read_df(t, tfmt, store, pathtoloc).to_numpy()
#     elif verbose:
#       print("Localization matrix not updated!")
#
#     tnew = times[i]
#     dt = int((tnew - t).days)
#     index = T_real[i] > 0
#     ypred = np.einsum('a,a', T_real[i,index], population[index]) / np.sum(population[index])
#
#     # root finding method
#     def func_root(b):
#       X_new = compute_sir_X(X, dt, b*B, gamma, method_solver)
#       T = 1.- X_new.reshape(2, N)[0]
#       y = np.einsum('a,a', T[index], population[index]) / np.sum(population[index])
#       # y = get_sir_omega_X(X_new, population)
#       # print("nnzero = {:d}".format(np.sum(np.int_(index))))
#       return y - ypred
#
#     # func_root = lambda b:  \
#     #     np.einsum('a,a', \
#     #     (1.- compute_sir_X(X, dt, b*B, gamma, method_solver).reshape(2, N)[0])[index], \
#     #     population[index]) / np.sum(population[index]) - ypred
#     # func_root = lambda b: get_sir_omega_X(compute_sir_X(X, dt, b*B, gamma, method_solver), \
#     #                                       population) - ypred
#
#     # initial bracketing
#     bhi = b_scale
#     fscale = 3.
#     for k in range(1,10):
#       f = func_root(bhi)
#       if f > 0:
#         break
#       else:
#         bhi *= fscale
#     if f < 0:
#       raise ValueError("Problem in bracketing!")
#
#     # find the root
#     sol = scipy.optimize.root_scalar(func_root, bracket=(blo, bhi), method='brentq', \
#                                       options={'maxiter': 100})
#     if not (sol.converged):
#       raise ValueError("root finding failed!")
#     b_scale = sol.root
#
#     # compute next state with optimal scale
#     t_eval = np.arange(dt+1)
#     Xnews = compute_sir_X(X, dt, b_scale*B, gamma, method_solver, t_eval=t_eval)
#     Xnew = Xnews[-1]
#     # y = get_sir_omega_X(Xnew,population)
#     T = 1.- Xnew.reshape(2, N)[0]
#     y = np.einsum('a,a', T[index], population[index]) / np.sum(population[index])
#     print(f"b = {b_scale}, y = {y}, ypred = {ypred}, y-ypred = {y-ypred}")
#
#     # dump
#     # data.append(Xnew.reshape(2,N))
#     Xs += [Xnew.reshape(2,N) for Xnew in Xnews]
#     ts += [t + datetime.timedelta(days=int(dt)) for dt in t_eval[1:]]
#     Ys.append(y)
#     b_scales.append(b_scale)
#
#     # update
#     t = tnew
#     X = Xnew
#
#   b_scales.append(None)  # B has ndays-1 entries
#   print("Fitting complete")
#
#   # prepare export of results
#   S = np.array([X[0] for X in Xs])
#   I = np.array([X[1] for X in Xs])
#   clusters = np.arange(N, dtype=np.uint)
#   df_S = pd.DataFrame(data=S, index=ts, columns=clusters)
#   df_I = pd.DataFrame(data=I, index=ts, columns=clusters)
#   df_fit = pd.DataFrame(data=np.array([b_scales, Ys]).T, index=times, columns=["localization_scale", "frac_infected_tot"])
#
#   return df_S, df_I, df_fit
#
# def fit_sir_vector(times, T_real, gamma, population, store, pathtomat, tfmt='%Y-%m-%d', method_solver='DOP853', verbose=True, \
#             b_scale=1, loss_KP=False):
#   """
#   Fit the dynamics of the SIR starting to real data contained in `pathtocssegi`.
#   The initial condition is taken from the real data.
#   The method assumes that in the `store` at the indicated `path`, there are entries
#   in the format %Y-%m-%d that described the flux matrices
#   for the times `times[:-1]`.
#   `populations` is the vector with the population per community.
#
#   OUTPUT:
#     * S
#     * I
#     * scales
#
#   For the output the dumping interval is one day.
#   """
#
#   # initializations
#   nt = len(times)
#   t = times[0]
#   F = read_df(t, tfmt, store, pathtomat).to_numpy()
#   N = F.shape[0]
#   # Y_real = np.einsum('ta,a->t', T_real[:,index], population[index]) / np.sum(population[index])
#
#   # X = np.zeros((2, N), dtype=np.float_)
#   I = T_real[0]
#   S = 1. - I
#   T = 1. - S
#   X = sir_SI_to_X(S, I)
#
#   y = np.einsum('a,a', T, population) / np.sum(population)
#   ypred = np.einsum('a,a', T_real[0], population) / np.sum(population)
#   print("y = {:.4e}    y_real = {:.4e}".format(y, ypred))
#   # y = get_sir_omega_X(X, population)
#
#   ts = [t]
#   Xs = [X.reshape(2,N)]
#   Ys = [0]
#   Vs = []
#
#   blo = 0.
#   # print("nt = ", nt)
#
#   #==========================================================================================
#   # first fit
#   #==========================================================================================
#   i = 1
#   mykey = Path(pathtomat) / t.strftime(tfmt)
#   mykey = str(mykey)
#   if mykey in store.keys():
#     F = read_df(t, tfmt, store, pathtomat).to_numpy()
#     B = get_localization_matrix(F)
#   elif verbose:
#     print("Localization matrix not updated!")
#
#   tnew = times[i]
#   dt = int((tnew - t).days)
#   index = T_real[i] > 0
#   ypred = np.einsum('a,a', T_real[i], population) / np.sum(population)
#
#   # root finding method
#   def func_root(b):
#     X_new = compute_sir_X(X, dt, b*B, gamma, method_solver)
#     T = 1.- X_new.reshape(2, N)[0]
#     y = np.einsum('a,a', T, population) / np.sum(population)
#     return y - ypred
#
#   # initial bracketing
#   bhi = b_scale
#   fscale = 3.
#   for k in range(1,10):
#     f = func_root(bhi)
#     if f > 0:
#       break
#     else:
#       bhi *= fscale
#   if f < 0:
#     raise ValueError("Problem in bracketing!")
#
#   # find the root
#   sol = scipy.optimize.root_scalar(func_root, bracket=(blo, bhi), method='brentq', \
#                                     options={'maxiter': 100})
#   if not (sol.converged):
#     raise ValueError("root finding failed!")
#   b_scale = sol.root
#
#   # compute next state with optimal scale
#   t_eval = np.arange(dt+1)
#   Xnews = compute_sir_X(X, dt, b_scale*B, gamma, method_solver, t_eval=t_eval)
#   Xnew = Xnews[-1]
#   T = 1.- Xnew.reshape(2, N)[0]
#   y = np.einsum('a,a', T, population) / np.sum(population)
#   print(f"b = {b_scale}, y = {y}, ypred = {ypred}, y-ypred = {y-ypred}")
#
#   def func_min(V):
#     B = get_localization_matrix_vscale(F,V)
#     X_new = compute_sir_X(X, dt, B, gamma, method_solver)
#     S_new, I_new = X_new.reshape(2, N)
#     T_new = 1. - S_new
#
#     if loss_KP:
#       # Kullback-Leibler distance
#       from scipy.special import xlogy
#       A = np.einsum('a,a->a', T_new, population)
#       B = np.einsum('a,a->a', T_real[i], population)
#       A /= np.sum(A)
#       B /= np.sum(B)
#       idx = (A > 0) & (B > 0)
#       A = A[idx]
#       B = B[idx]
#       return np.sum((A-B)*(np.log(A) - np.log(B)))
#
#     else:
#       # least-square difference
#       return np.sum(((T_new-T_real[i])*population)**2)/N  # take this one
#
#
#   def func_min_jac(V):
#     """
#     Return the jacobian of the matrix to minimize
#     """
#     B = get_localization_matrix_vscale(F,V)
#     S, I = X.reshape(2, N)
#     X_new = compute_sir_X(X, dt, B, gamma, method_solver)
#     S_new, I_new = X_new.reshape(2, N)
#     T_new = 1. - S_new
#
#     if loss_KP:
#       print("KP")
#       # Kullback-Leibler distance
#       A = np.einsum('a,a->a', T_new, population)
#       B = np.einsum('a,a->a', T_real[i], population)
#       Z_A = np.sum(A)
#       Z_B = np.sum(B)
#       A = A/Z_A
#       B = B/Z_B
#       idx = (A > 0) & (B > 0)
#
#       J1 = S_new*(0.5*I+0.5*I_new)*dt*population/Z_A
#
#       J2 = np.zeros(N, dtype=np.float_)
#       J2[idx] = - B[idx]/A[idx] + np.log(A[idx]) - np.log(B[idx])
#       Z = np.einsum('a,a', J2, A)
#       J2 = J2 - Z
#       J = J1*J2
#
#     else:
#       print("LSQ")
#       # least-square difference
#       J = 2.*S_new*(0.5*I+0.5*I_new)*dt*(T_new-T_real[i])*population**2 / N
#
#     return J
#
#   ## CHECK GRADIENT
#   # print("CHECKING GRADIENT")
#   # V_init = np.ones(N, dtype=np.float_)*b_scale
#   # itermax=50
#   # YY = []
#   #
#   # V = V_init.copy()
#   # f = func_min(V)
#   # G = func_min_jac(V)
#   # # G = np.random.rand(N)
#   # alpha=1.0e0
#   # YY.append(f)
#   # for it in range(itermax):
#   #   print(it, f)
#   #   V = V - alpha*G
#   #
#   #   f = func_min(V)
#   #   G = func_min_jac(V)
#   #   YY.append(f)
#   #
#   # TEST
#   # B = get_localization_matrix_vscale(F,V)
#   # S, I = X.reshape(2, N)
#   # X_new = compute_sir_X(X, dt, B, gamma, method_solver)
#   # S_new, I_new = X_new.reshape(2, N)
#   # T_new = 1. - S_new
#   # A = np.einsum('a,a->a', T_new, population)
#   # B = np.einsum('a,a->a', T_real[i], population)
#   # Z_A = np.sum(A)
#   # Z_B = np.sum(B)
#   # A = A/Z_A
#   # B = B/Z_B
#   # idx = (A > 0) & (B > 0)
#   #
#   # J1 = S_new*(0.5*I+0.5*I_new)*dt*population/Z_A
#   #
#   # J2 = np.zeros(N, dtype=np.float_)
#   # J2[idx] = - B[idx]/A[idx] + np.log(A[idx]) - np.log(B[idx])
#   # Z = np.einsum('a,a', J2, A)
#   # J2 = J2 - Z
#   # return np.arange(itermax+1), np.array(YY), G, A*Z_A, B*Z_B, J1, J2
#   # TEST
#   # return np.arange(itermax+1), np.array(YY)
#
#   from scipy.optimize import minimize
#   # CLASSICAL MINIMIZATION ROUTING
#   method = 'L-BFGS-B'
#   # method = 'Nelder-Mead'
#   bounds = [(0., 1.)]*N
#   # def print_fun(x):
#   #   print("xmin = {:.6f}    xmax = {:.6f}".format(np.min(x), np.max(x)))
#   print_fun=None
#
#   V = np.ones(N, dtype=np.float_)*b_scale
#   sol = minimize(func_min, V, jac=func_min_jac, \
#       method=method, bounds=bounds, callback=print_fun)
#   Vnew = sol.x
#
#   ## BASINHOPPING
#   # from scipy.optimize import basinhopping
#   # class MyBounds:
#   #   def __init__(self, xmin=[0.0]*N, xmax=[1.0]*N):
#   #     self.xmax = np.array(xmax)
#   #     self.xmin = np.array(xmin)
#   #   def __call__(self, **kwargs):
#   #     x = kwargs["x_new"]
#   #     tmax = bool(np.all(x <= self.xmax))
#   #     tmin = bool(np.all(x >= self.xmin))
#   #     return tmax and tmin
#   #
#   # mybounds = MyBounds()
#   #
#   # def print_fun(x, f, accepted):
#   #   print("at minimum %.4f accepted %d" % (f, int(accepted)))
#   #
#   # V = np.ones(N, dtype=np.float_)*b_scale
#   # # np.random.seed(123)
#   # # V = np.random.rand(N)
#   # # np.random.seed(123)
#   # # V = np.random.rand(N)*b_scale
#   # # V = np.zeros(N)
#   # sol = basinhopping(func_min, V, \
#   #     niter=200, T=1.0e-4, stepsize=0.05, \
#   #     minimizer_kwargs={"method": "L-BFGS-B", "jac": func_min_jac, "bounds": [(0., 1.)]*N}, \
#   #     callback=print_fun, \
#   #     accept_test=mybounds)
#   # Vnew = sol.x
#
#   # compute solution at optimum
#   B = get_localization_matrix_vscale(F,Vnew)
#   t_eval = np.arange(dt+1)
#   Xnews = compute_sir_X(X, dt, B, gamma, method_solver, t_eval=t_eval)
#   Xnew = Xnews[-1]
#   y = func_min(Vnew)
#   print("f = {:.6e}".format(y))
#
#   Xs += [Xnew.reshape(2,N) for Xnew in Xnews]
#   ts += [t + datetime.timedelta(days=int(dt)) for dt in t_eval[1:]]
#   Ys.append(y)
#   Vs.append(Vnew)
#
#   # update
#   t = tnew
#   X = Xnew
#   V = Vnew
#
#   S_new, I_new = Xnew.reshape(2,N)
#   T_new = 1 - S_new
#   # return sol, Vnew, T_new, T_real[i]
#
#   #==========================================================================================
#   # after first fit
#   #==========================================================================================
#   for i in range(2, nt):
#     if verbose:
#       print(f'Integrating day {t}')
#     mykey = Path(pathtomat) / t.strftime(tfmt)
#     mykey = str(mykey)
#     if mykey in store.keys():
#       F = read_df(t, tfmt, store, pathtomat).to_numpy()
#       B = get_localization_matrix(F)
#     elif verbose:
#       print("Localization matrix not updated!")
#
#     tnew = times[i]
#     dt = int((tnew - t).days)
#
#     def func_min(V):
#       B = get_localization_matrix_vscale(F,V)
#       X_new = compute_sir_X(X, dt, B, gamma, method_solver)
#       S_new, I_new = X_new.reshape(2, N)
#       T_new = 1. - S_new
#
#       if loss_KP:
#         # Kullback-Leibler distance
#         from scipy.special import xlogy
#         A = np.einsum('a,a->a', T_new, population)
#         B = np.einsum('a,a->a', T_real[i], population)
#         A /= np.sum(A)
#         B /= np.sum(B)
#         idx = (A > 0) & (B > 0)
#         A = A[idx]
#         B = B[idx]
#         return np.sum((A-B)*(np.log(A) - np.log(B)))
#
#       else:
#         # least-square difference
#         return np.sum(((T_new-T_real[i])*population)**2)/N  # take this one
#
#
#     def func_min_jac(V):
#       """
#       Return the jacobian of the matrix to minimize
#       """
#       B = get_localization_matrix_vscale(F,V)
#       S, I = X.reshape(2, N)
#       X_new = compute_sir_X(X, dt, B, gamma, method_solver)
#       S_new, I_new = X_new.reshape(2, N)
#       T_new = 1. - S_new
#
#       if loss_KP:
#         # Kullback-Leibler distance
#         A = np.einsum('a,a->a', T_new, population)
#         B = np.einsum('a,a->a', T_real[i], population)
#         Z_A = np.sum(A)
#         Z_B = np.sum(B)
#         A = A/Z_A
#         B = B/Z_B
#         idx = (A > 0) & (B > 0)
#
#         J1 = S_new*(0.5*I+0.5*I_new)*dt*population/Z_A
#
#         J2 = np.zeros(N, dtype=np.float_)
#         J2[idx] = - B[idx]/A[idx] + np.log(A[idx]) - np.log(B[idx])
#         Z = np.einsum('a,a', J2, A)
#         J2 = J2 - Z
#         J = J1*J2
#
#       else:
#         # least-square difference
#         J = 2.*S_new*(0.5*I+0.5*I_new)*dt*(T_new-T_real[i])*population**2 / N
#
#       return J
#
#     ## CLASSICAL MINIMIZATION ROUTINGE
#     from scipy.optimize import minimize
#     method = 'L-BFGS-B'
#     # method = 'Nelder-Mead'
#     bounds = [(0., 1.)]*N
#     # def print_fun(x):
#     #   print("xmin = {:.6f}    xmax = {:.6f}".format(np.min(x), np.max(x)))
#     print_fun=None
#
#     sol = minimize(func_min, V, jac=func_min_jac, \
#         method=method, bounds=bounds)
#     Vnew = sol.x
#
#     B = get_localization_matrix_vscale(F,Vnew)
#     t_eval = np.arange(dt+1)
#     Xnews = compute_sir_X(X, dt, B, gamma, method_solver, t_eval=t_eval)
#     Xnew = Xnews[-1]
#     y = func_min(Vnew)
#     print("f = {:.6e}".format(y))
#
#     Xs += [Xnew.reshape(2,N) for Xnew in Xnews]
#     ts += [t + datetime.timedelta(days=int(dt)) for dt in t_eval[1:]]
#     Ys.append(y)
#     Vs.append(Vnew)
#
#     # update
#     t = tnew
#     X = Xnew
#     V = Vnew
#
#   print("Fitting complete")
#
#   # prepare export of results
#   S = np.array([X[0] for X in Xs])
#   I = np.array([X[1] for X in Xs])
#   clusters = np.arange(N, dtype=np.uint)
#   df_S = pd.DataFrame(data=S, index=ts, columns=clusters)
#   df_I = pd.DataFrame(data=I, index=ts, columns=clusters)
#   df_scales = pd.DataFrame(data=np.array(Vs), index=times[:-1], columns=clusters)
#   df_fit = pd.DataFrame(data=np.array(Ys), index=times, columns=["f"])
#
#   return df_S, df_I, df_scales, df_fit
# #==============================================================================
# # plot methods
# #==============================================================================
# def autolabel_vertical(ax, rects, fontsize='small', fmt_str='{:.2f}'):
#     """
#     Attach a text label above each bar in *rects*, displaying its height.
#     From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
#     """
#
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(fmt_str.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=fontsize)
#
# def autolabel_horizontal(ax, rects, fontsize='small', fmt_str='{:.2f}'):
#     """
#     Attach a text label to the right each bar in *rects*, displaying its width.
#     """
#
#     for rect in rects:
#         width = rect.get_width()
#         ax.annotate(fmt_str.format(width),
#                     xy=(width, rect.get_y() + rect.get_height() / 2),
#                     xytext=(3, 0),  # 3 points horizontal offset
#                     textcoords="offset points",
#                     ha='left', va='center', fontsize=fontsize)
#
#
# def plot_xy(X,Y, lw=0.5, ms=2, figsize=(4,3)):
#     """
#     Example of method
#     """
#
#     fig = plt.figure(facecolor='w', figsize=figsize)
#     ax = fig.gca()
#
#     ax.plot(X,Y, '-', lw=lw, ms=ms)
#
#     ax.set_xlabel("x", fontsize="medium")
#     ax.set_ylabel("y", fontsize="medium")
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.tick_params(left=True, labelleft=True, bottom=True, labelbottom=True)
#     ax.tick_params(axis='both', length=4)
#     fig.tight_layout()
#
#     return
#
def show_image(mat_, downscale=None, log=False, mpl=False, vmin=None, vmax=None, fileout=None, dpi=72, interpolation='none', method='sum'):
    mat = np.copy(mat_)
    N = mat.shape[0]
    if downscale:
        NK = N // downscale
        if method == 'sum':
          mat = mat[:NK*downscale, :NK*downscale].reshape(NK, downscale, NK, downscale).sum(axis=(1, 3))
        elif method == 'max':
          mat = mat[:NK*downscale, :NK*downscale].reshape(NK, downscale, NK, downscale).max(axis=(1, 3))
        elif method == 'maxmin':
          mat1 = mat[:NK*downscale, :NK*downscale].reshape(NK, downscale, NK, downscale).max(axis=(1, 3))
          mat12 = np.abs(mat1)
          mat2 = mat[:NK*downscale, :NK*downscale].reshape(NK, downscale, NK, downscale).min(axis=(1, 3))
          mat22 = np.abs(mat2)
          theta = np.int_(mat22 > mat12)
          mat = (1.-theta)*mat1 + theta*mat2

        else:
          raise ValueError("Method not implemented!")

    if not mpl:
        if log:
            mat = np.log(mat)

        fig = px.imshow(mat)
        return fig
        # fig.show()

    else:
        fig = plt.figure()
        ax = fig.gca()
        if log:
            img = ax.imshow(mat, norm=mco.LogNorm(vmin=vmin, vmax=vmax), extent=[0,N-1,N-1,0], origin='upper', interpolation=interpolation)
        else:
            img = ax.imshow(mat, origin='upper', interpolation=interpolation)

        plt.colorbar(img)
        if fileout:
          fig.savefig(fileout, dpi=dpi, bbox_inches='tight', pad_inches=0)
          print("Written file {:s}".format(str(fileout)))
          fig.clf()
          plt.close('all')
        else:
          return fig
          # plt.show()

# def plot_v_profile(Xs, Ys, labels=None, colors=None, fileout=None, dpi=150, xmin=None, xmax=None, ymin=None, ymax=None, \
#     figsize=(4,3), xlabel=None, ylabel=None, lws=None, ms=2):
#   """
#   Plot (X,Y) as a series of curves on the same plot.
#   """
#   ndata = len(Xs)
#   if (ndata == 0):
#     raise ValueError("Empty data!")
#   if (len(Ys) != ndata):
#     raise ValueError("Xs and Ys must have same lengths!")
#
#   fig = plt.figure(figsize=figsize, dpi=dpi)
#   ax = fig.gca()
#   ax.spines['right'].set_visible(False)
#   ax.spines['top'].set_visible(False)
#   ax.set_xlabel(xlabel, fontsize='large')
#   ax.set_ylabel(ylabel, fontsize='large')
#
#   if labels is None:
#     labels =[None]*ndata
#   haslabel=False
#   for label in labels:
#     if label:
#       haslabel=True
#       break
#   if lws is None:
#     lws = [0.5]*ndata
#
#   if colors is None:
#     colors =[None]*ndata
#
#   for i in range(ndata):
#     X = Xs[i]
#     Y = Ys[i]
#     color = colors[i]
#     label = labels[i]
#     lw = lws[i]
#
#     ax.plot(X, Y,'-', lw=lw, color=color, label=label)
#
#   if haslabel:
#     ax.legend(loc='best', fontsize='medium')
#
#   ax.set_xlim(xmin, xmax)
#   ax.set_ylim(ymin, ymax)
#   ax.tick_params(length=4)
#
#   fig.tight_layout()
#   if fileout:
#     fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
#     print("Written file {:s}".format(str(fileout)))
#     fig.clf()
#     plt.close('all')
#     return
#   else:
#     return fig
#
# def plot_omega_profile_old(df_tolls, labels=None, colors=None, fileout=Path('./animation.gif'), dpi=150, ymin=None, ymax=None, figsize=(4,3), nframes=None, fps=5, \
#     log=True, ylabel="$\Omega_a$", lgd_ncol=2):
#   """
#   Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
#   representing the dynamics of local epidemic sizes
#   See this tutorial on how to make animated movies:
#     https://matplotlib.org/stable/api/animation_api.html
#   INPUT:
#     * df_tolls: list of dataframes
#   """
#
#   nseries = len(df_tolls)
#   fig = plt.figure(figsize=figsize, dpi=dpi)
#   ax = fig.gca()
#   ax.spines['right'].set_visible(False)
#   ax.spines['top'].set_visible(False)
#
#   # determine minimum
#   # ymin = 10**np.floor(np.log10(1/pops.max()))
#   if ymin is None:
#     X = df_tolls[0].iloc[0].to_numpy()
#     ymin = 10**(np.floor(np.log10(np.min(X[X>0.]))))    # closest power of 10
#   if ymax is None:
#     X = df_tolls[0].iloc[-1].to_numpy()
#     ymax = 10**(np.ceil(np.log10(np.max(X[X>0.]))))    # closest power of 10
#   print("ymin = {:.2e}".format(ymin), "ymax = {:.2e}".format(ymax))
#   ax.set_ylim(ymin,ymax)
#
#   X = df_tolls[0].columns.to_numpy()
#   ax.set_xlim(0., X[-1])
#   if log:
#     ax.set_yscale('log')
#   ax.set_xlabel('cluster', fontsize='medium')
#   ax.set_ylabel(ylabel, fontsize='large')
#   # colors=px.colors.qualitative.Plotly[:nseries]
#
#   haslabels=True
#   if labels is None:
#     haslabels=False
#     labels = [None]*nseries
#
#   if colors is None:
#     colors = [None]*nseries
#
#   arrs = []
#   txt = ax.set_title("tp", fontsize='large')
#
#
#   def animate(i, arrs, txt, haslabels):
#     for arr in arrs:
#       if arr:
#         arr.remove()
#
#     arrs.clear()
#
#     t = df_tolls[0].index[i]
# #     Y1 = data_real_x.loc[t].to_numpy()
#     for k in range(nseries):
#       color = colors[k]
#       Y = df_tolls[k].loc[t].to_numpy()
#       arr, = ax.plot(X, Y, '-', color=color, lw=0.5)
#       arrs.append(arr)
#
#     date = t.strftime('%Y-%m-%d')
#     title = "{:s}".format(date)
#     txt.set_text(title)
#
#     if haslabels:
#       # ax.legend(loc='upper left', fontsize='xx-small', frameon=False, ncol=2)
#       ax.legend(arrs, labels=labels, loc='upper left', fontsize='xx-small', frameon=False, ncol=lgd_ncol)
#     return
#
#   def init():
#     return
#
#   if nframes is None:
#     nframes=len(df_tolls[0])
#     print("nframes = {:d}".format(nframes))
#
#   anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes, blit=False, fargs=(arrs, txt, haslabels))
#   anim.save(fileout, fps=fps)
#   print("Written file {:s}.".format(str(fileout)))
#   return
#
# def plot_omega_profile(Omegas, times, labels=None, colors=None, styles=None, fileout=Path('./animation.gif'), tpdir=Path('.'), \
#                        dpi=150, lw=0.5, ms=2, idump=10, \
#                        ymin=None, ymax=None, figsize=(4,3), fps=5, \
#                        log=True, xlabel='community', ylabel="$\Omega_a$", lgd_ncol=2, deletetp=True, exts=['.png'], \
#                        tfmt = "%Y-%m-%d"):
#   """
#   Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
#   representing the dynamics of local epidemic sizes
#   See this tutorial on how to make animated movies:
#     https://matplotlib.org/stable/api/animation_api.html
#   INPUT:
#     * Omegas: list of table containing omegas (indices t,a)
#     * times: list of times (indices t)
#   """
#   # tp dir
#   if not tpdir.is_dir():
#     tpdir.mkdir(exist_ok=True)
#   for ext in exts:
#     for f in tpdir.glob('*' + ext): f.unlink()
#
#   # parameters
#   nseries = Omegas.shape[0]
#   nt = len(times)
#   if (Omegas.shape[1] != nt):
#     raise ValueError("Omegas must have same second dimension as times!")
#   N = Omegas.shape[2]
#
#   haslabels=True
#   if labels is None:
#     haslabels=False
#     labels = [None]*nseries
#
#   if colors is None:
#     colors = [None]*nseries
#
#   if styles is None:
#     styles = [None]*nseries
#   for k in range(nseries):
#     if styles[k] is None:
#       styles[k] = 'o'
#
#   num = int(np.ceil(np.log10(nt)))
#   if float(nt) == float(10**num):
#     num += 1
#   # fmt = "{" + ":0{:d}".format(num) + "}"
#
#   # determine minimum and maximum
#   idx = Omegas[:,0,:] > 0.
#   if ymin is None:
#     ymin = 10**(np.floor(np.log10(np.min(Omegas[:,0,:][idx]))))    # closest power of 10
#   if ymax is None:
#     ymax = 10**(np.ceil(np.log10(np.max(Omegas))))    # closest power of 10
#   print("ymin = {:.2e}".format(ymin), "ymax = {:.2e}".format(ymax))
#
#   if not ".png" in exts:
#     raise ValueError("PNG format must be given")
#
#   # community index
#   X = np.arange(N, dtype=np.uint)
#
#   # prepare figure
#   filenames=[]
#   for i in range(nt):
#     ## update time and Omega
#     t = times[i]
#
#     ## create figure
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     ax = fig.gca()
#
#     date = t.strftime('%Y-%m-%d')
#     title = "{:s}".format(date)
#     ax.set_title(title, fontsize="large")
#
#     for k in range(nseries):
#       Y = Omegas[k, i]
#       color=colors[k]
#       label=labels[k]
#       style=styles[k]
#       ax.plot(X, Y, style, lw=lw, mew=0, ms=ms, color=color, label=label)
#
#     ax.set_xlim(X[0], X[-1])
#     ax.set_ylim(ymin,ymax)
#     ax.set_xlabel(xlabel, fontsize='medium')
#     ax.set_ylabel(ylabel, fontsize='large')
#     if log:
#       ax.set_yscale('log')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.tick_params(length=4)
#
#     if haslabels:
#       ax.legend(loc='lower left', fontsize='medium', frameon=False, ncol=lgd_ncol)
#
#     fname = str(tpdir / t.strftime(tfmt))
#     # fname = str(tpdir / fmt.format(i))
#     for ext in exts:
#       fpath = fname + ext
#       fig.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
#     fpath = fname + ".png"
#     filenames.append(fpath)
#
#     if (i %idump == 0):
#       print(f"Written file {fpath}.")
#
#     fig.clf()
#     plt.close('all')
#
#   # write movie
#   imageio.mimsave(fileout, [imageio.imread(f) for f in filenames], fps=fps)
#   print(f"Written file {fileout}.")
#
#   # clean tpdir
#   if deletetp:
#     shutil.rmtree(tpdir)
#
#   return
#
# def plot_omega_profile_two(Y1, Y2, times, labels=None, colors=None, fileout=Path('./animation.gif'), tpdir=Path('.'), \
#                        dpi=150, lw=0.5, ms=2, idump=10, \
#                        ymin=None, ymax=None, figsize=(4,3), fps=5, \
#                        log=True, xlabel='community', ylabel="$\Omega_a$", lgd_ncol=2, deletetp=True, exts=['.png'], \
#                        tfmt = "%Y-%m-%d"):
#   """
#   Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
#   representing the dynamics of local epidemic sizes
#   See this tutorial on how to make animated movies:
#     https://matplotlib.org/stable/api/animation_api.html
#   INPUT:
#     * Omegas: list of table containing omegas (indices t,a)
#     * times: list of times (indices t)
#   """
#   # tp dir
#   if not tpdir.is_dir():
#     tpdir.mkdir(exist_ok=True)
#   for ext in exts:
#     for f in tpdir.glob('*' + ext): f.unlink()
#
#   # parameters
#   nseries = 2
#   nt = len(times)
#   if (Y1.shape[0] != nt):
#     raise ValueError("Y1 must have same first dimension as times!")
#   if (Y2.shape[0] != nt):
#     raise ValueError("Y2 must have same first dimension as times!")
#   N = Y1.shape[1]
#   if (Y2.shape[1] != N):
#     raise ValueError("Y2 must have same second dimension as Y1!")
#
#   haslabels=True
#   if labels is None:
#     haslabels=False
#     labels = [None]*nseries
#
#   if colors is None:
#     colors = [None]*nseries
#
#   # num = int(np.ceil(np.log10(nt)))
#   # if float(nt) == float(10**num):
#   #   num += 1
#   # fmt = "{" + ":0{:d}".format(num) + "}"
#
#   # determine minimum and maximum
#   Y_all = np.array([Y1, Y2])
#   idx = Y_all[:,0,:] > 0.
#   if ymin is None:
#     ymin = 10**(np.floor(np.log10(np.min(Y_all[:,0,:][idx]))))    # closest power of 10
#   if ymax is None:
#     ymax = 10**(np.ceil(np.log10(np.max(Y_all))))    # closest power of 10
#   print("ymin = {:.2e}".format(ymin), "ymax = {:.2e}".format(ymax))
#
#   if not ".png" in exts:
#     raise ValueError("PNG format must be given")
#
#   # community index
#   X = np.arange(N, dtype=np.uint)
#   edges = np.arange(N+1, dtype=np.float_)
#
#   # prepare figure
#   filenames=[]
#   for i in range(nt):
#     ## update time and Omega
#     t = times[i]
#
#     ## create figure
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     ax = fig.gca()
#
#     date = t.strftime('%Y-%m-%d')
#     title = "{:s}".format(date)
#     ax.set_title(title, fontsize="large")
#
#     for k in range(nseries):
#       idx = Y1[i]>=ymin
#       ax.bar(edges[:-1][idx], Y1[i][idx], np.diff(edges)[0], lw=0, color=colors[0], ec='k', label=labels[0])
#       idx = Y2[i]>=ymin
#       ax.plot(X[idx], Y2[i][idx], 'o', lw=lw, mew=0, ms=ms, color=colors[1], label=labels[1])
#
#     ax.set_xlim(edges[0], edges[-1])
#     ax.set_ylim(ymin,ymax)
#     ax.set_xlabel(xlabel, fontsize='medium')
#     ax.set_ylabel(ylabel, fontsize='large')
#     if log:
#       ax.set_yscale('log')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.tick_params(length=4)
#
#     if haslabels:
#       ax.legend(loc='lower left', fontsize='medium', frameon=False, ncol=lgd_ncol)
#
#     fname = str(tpdir / t.strftime(tfmt))
#     # fname = str(tpdir / fmt.format(i))
#     for ext in exts:
#       fpath = fname + ext
#       fig.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
#     fpath = fname + ".png"
#     filenames.append(fpath)
#
#     if (i %idump == 0):
#       print(f"Written file {fpath}.")
#
#     fig.clf()
#     plt.close('all')
#
#   # write movie
#   imageio.mimsave(fileout, [imageio.imread(f) for f in filenames], fps=fps)
#   print(f"Written file {fileout}.")
#
#   # clean tpdir
#   if deletetp:
#     shutil.rmtree(tpdir)
#
#   return
#
# def plot_omega_map(Omega, times, XY, fileout=Path('./animation.gif'), tpdir=Path('.'), dpi=150, \
#                    vmin=None, vmax=None, figsize=(4,3), nframes=None, fps=5, \
#                    cmap=cm.magma_r, idump=1, tfmt = "%Y-%m-%d", ymin=None, ymax=None, \
#                    clabel='$\Omega$', deletetp=True, exts=['.png'], \
#                    circle_size=0.4, lw=0.1, edges=[], edge_width=0.5):
#   """
#   Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
#   representing the dynamics of local epidemic sizes
#
#   INPUT:
#     * df_tolls: list of dataframes
#     * XY: 2xN array giving the coordinates of the N communities.
#     *
#   """
#   from matplotlib.path import Path
#   # tp dir
#   if not tpdir.is_dir():
#     tpdir.mkdir(exist_ok=True)
#   for ext in exts:
#     for f in tpdir.glob('*' + ext): f.unlink()
#
#   # parameters
#   nt = len(times)
#   if (Omega.shape[0] != nt):
#     raise ValueError("Omega must have same second dimension as times!")
#   N = Omega.shape[1]
#
#   num = int(np.ceil(np.log10(nt)))
#   if float(nt) == float(10**num):
#     num += 1
#   fmt = "{" + ":0{:d}".format(num) + "}"
#
#   # color scale
#   # determine minimum and maximum
#   idx = Omega[0,:] > 0.
#   if vmin is None:
#     vmin = 10**(np.floor(np.log10(np.min(Omega[0,:][idx]))))    # closest power of 10
#   if vmax is None:
#     vmax = 10**(np.ceil(np.log10(np.max(Omega))))    # closest power of 10
#   print("vmin = {:.2e}".format(vmin), "vmax = {:.2e}".format(vmax))
#   norm = mco.LogNorm(vmin=vmin, vmax=vmax)
#
#   # clusters
#   X, Y = XY
#   xmin = np.min(X)
#   xmax = np.max(X)
#   if ymin is None:
#     ymin = np.min(Y)
#   if ymax is None:
#     ymax = np.max(Y)
#
#   # prepare figure
#   filenames=[]
#   for i in range(nt):
#     if (i %idump != 0):
#       continue
#     ## update time and Omega
#     t = times[i]
#
#     ## create figure
#     fig = plt.figure(figsize=figsize, dpi=dpi)
#     ax = fig.gca()
#
#     date = t.strftime('%Y-%m-%d')
#     title = "{:s}".format(date)
#     ax.set_title(title, fontsize="large")
#
#     # draw edges
#     if len(edges) > 0:
#       for a1,a2 in edges:
#         x1 = X[a1]
#         y1 = Y[a1]
#         x2 = X[a2]
#         y2 = Y[a2]
#         # ax.plot([x1,x2], [y1,y2], 'k-', lw=edge_width)
#         verts = [ (x1, y1), (x2, y2)]
#         codes = [Path.MOVETO, Path.LINETO]
#         path = Path(verts, codes)
#         patch = mpatches.PathPatch(path, facecolor='none', edgecolor='k', lw=edge_width)
#         res = ax.add_patch(patch)
#
#     # draw spheres
#     Ns = np.arange(N)
#     idx = np.argsort(Omega[i])
#     # for a in range(N):
#     for a in Ns[idx]:
#       x = X[a]
#       y = Y[a]
#       val = Omega[i,a]
#       if (val < vmin):
#         color = [1.,1.,1.,1.]
#       elif (val > vmax):
#         color = [0.,0.,0.,1.]
#       else:
#         color = cmap(norm(val))
#       circle = plt.Circle((x,y), circle_size, color=color, alpha=1, lw=lw, ec='black')
#       res = ax.add_patch(circle)
#
#     # formatting
#     for lab in 'left', 'right', 'bottom', 'top':
#       ax.spines[lab].set_visible(False)
#     ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
#     cax = fig.add_axes(rect=[0.98,0.1,0.02,0.7])
#     plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=clabel, extendfrac='auto')
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax.set_aspect('equal')
#
#     # write figure
#     fname = str(tpdir / t.strftime(tfmt))
#     for ext in exts:
#       fpath = fname + ext
#       fig.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
#     fpath = fname + ".png"
#     filenames.append(fpath)
#
#     print(f"Written file {fpath}.")
#
#     fig.clf()
#     plt.close('all')
#
#   # write movie
#   imageio.mimsave(fileout, [imageio.imread(f) for f in filenames], fps=fps)
#   print(f"Written file {fileout}.")
#
#   # clean tpdir
#   if deletetp:
#     shutil.rmtree(tpdir)
#
#   return
# def plot_omega_map_old(df_toll, XY, labels=None, fileout=Path('./animation.gif'), tpdir=Path('.'), dpi=150, vmin=None, vmax=None, figsize=(4,3), nframes=None, fps=5, \
#                    cmap=cm.magma_r, idump=1):
#   """
#   Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
#   representing the dynamics of local epidemic sizes
#
#   INPUT:
#     * df_tolls: list of dataframes
#     * XY: 2xN array giving the coordinates of the N communities.
#     *
#   """
#   # tp dir
#   if not tpdir.is_dir():
#     tpdir.mkdir(exist_ok=True)
#   for f in tpdir.glob('*.png'): f.unlink()
#
#   # color scale
#   if vmin is None:
#     X = df_toll.iloc[0].to_numpy()
#     vmin = 10**(np.floor(np.log10(np.min(X[X>0.]))))    # closest power of 10
#   if vmax is None:
#     X = df_toll.iloc[-1].to_numpy()
#     vmax = 10**(np.ceil(np.log10(np.max(X[X>0.]))))    # closest power of 10
#   print("vmin = {:.2e}".format(vmin), "vmax = {:.2e}".format(vmax))
#
#   norm = mco.LogNorm(vmin=vmin, vmax=vmax)
#
#   # clusters
#   X, Y = XY
#   xmin = np.min(X)
#   xmax = np.max(X)
#   ymin = np.min(Y)
#   ymax = np.max(Y)
#
#   # prepare figure
#   fig = plt.figure(figsize=figsize, dpi=dpi)
#   ax = fig.gca()
#   for lab in 'left', 'right', 'bottom', 'top':
#     ax.spines[lab].set_visible(False)
#   ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
#   cax = fig.add_axes(rect=[0.98,0.1,0.02,0.7])
#   plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='$\Omega$', extendfrac='auto')
#   ax.set_xlim(xmin, xmax)
#   ax.set_ylim(ymin, ymax)
#   ax.set_aspect('equal')
#
#   data = df_toll.to_numpy()
#   P = len(data)     # number of time points
#   N = len(X)        # number of communities
#   num = int(np.ceil(np.log10(P)))
#   if float(P) == float(10**num):
#     num += 1
#   fmt = "{" + ":0{:d}".format(num) + "}"
#
#   collection = []
#
#   if nframes is None:
#     nframes = P
#
#   filenames = []
#   for i in range(nframes):
#     # if (i %idump == 0):
#     #   print("frame {:d} / {:d}".format(i+1, nframes))
#
#     # delete existing collection
#     [c.remove() for c in collection]
#     collection = []
#
#     # draw artists
#     for a in range(N):
#       x = X[a]
#       y = Y[a]
#       val = data[i,a]
#       if (val < vmin):
#         color = [1.,1.,1.,1.]
#       elif (val > vmax):
#         color = [0.,0.,0.,0.]
#       else:
#         color = cmap(norm(val))
#       circle = plt.Circle((x,y), 0.4, color=color, alpha=1, lw=0.1, ec='black')
#       res = ax.add_patch(circle)
#       collection.append(res)
#
#     date = df_toll.index[i].strftime('%Y-%m-%d')
#     title = "{:s}".format(date)
#     ax.set_title(title, fontsize='large')
#
#     fig.canvas.draw()
#     fpath = str(tpdir / fmt.format(i))  + '.png'
#     fig.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
#     filenames.append(fpath)
#
#     if (i %idump == 0):
#       print(f"Written file {fpath}.")
#
#   # write movie
#   imageio.mimsave(fileout, [imageio.imread(f) for f in filenames], fps=fps)
#   print(f"Written file {fileout}.")
#
#   # clean tpdir
#   shutil.rmtree(tpdir)
#
#   return fig
#
# def wave_front_get_ode_sol(C, D=0, p0=-0.99, tmin=0, tmax=1000, npts=1000, t_eval=None, eps=1.0e-3, method='BDF', x0_inf=1.0e-12):
#     from scipy.integrate import solve_ivp
#
#     # first case: D = 0 (no recovery rate)
#     if D == 0:
#       def func_ode(t, x, *f_args):
#         """
#         function to integrate
#         """
#         C = f_args[0]
#         q, p = x
#
#         return np.array([-q/(1.+p) + C*(1.-p), q,])
#
#       def event_upperbound(t, x, *f_args):
#           return x[1]+p0
#
#       event_upperbound.terminal=True
#
#       q0 = C*(1-p0**2)
#       X0=np.array([q0,p0])
#       args = [C]
#
#       if t_eval is None:
#         t_eval = np.linspace(tmin,tmax,npts)
#
#       sol = solve_ivp(func_ode, t_span=[tmin,tmax], y0=X0, method=method, args=args, \
#           events=event_upperbound, t_eval=t_eval)
#
#       T = sol.t
#       X = sol.y[0]
#       Y = sol.y[1]
#       return T, X, Y
#
#     # second case: D > 0 (with recovery rate)
#     else:
#       def func_ode(t, X, *f_args):
#         """
#         function to integrate
#         """
#         C = f_args[0]
#         D = f_args[1]
#         CD = C*D
#         x, y, z = X
#
#         return np.array([-(1./(1.+y) + CD)*x  + C*(1+D*CD)*(z-y), x, CD*(z-y)])
#
#       def event_upperbound(t, X, *f_args):
# #         return x[1]-1.0
#         # return X[0]
#         return np.abs(X[0]) - x0_inf
#
#       event_upperbound.terminal=True
#
#       def get_final_state(y0, tmax=10000, method=method):
#         z0 = y0 + 2*eps # so that S+I+R=1
#         x0 = 2*C*eps*(1.+y0)
#         X0=np.array([x0,y0,z0])
#         args = [C, D]
#
#         sol = solve_ivp(func_ode, t_span=[0.,tmax], y0=X0, method=method, args=args, events=event_upperbound)
#         return sol.y[1,-1]
#
#       from scipy.optimize import root_scalar
#       func_min = lambda y: get_final_state(y) - 1.
#       delta=0.01
#       for it in range(6):
#         ylo = -1+delta
#         flo = func_min(ylo)
#         if flo > 0:
#           break
#         else:
#           delta /= 10
#       if (flo < 0.):
#         raise ValueError("flo < 0: change eps or tmax (most likely trajectory truncated early).")
#       yhi = D-1
#       fhi = func_min(yhi)
#       if (fhi > 0.):
#         raise ValueError("fhi > 0: change eps or tmax")
#
#       rt = root_scalar(func_min, method='brentq', bracket=[ylo,yhi])
#       y0 = rt.root
#
#       z0 = y0 + 2*eps # so that S+I+R=1
#       x0 = 2*C*eps*(1.+y0)
#       X0=np.array([x0,y0,z0])
#       args = [C, D]
#
#       if t_eval is None:
#         t_eval = np.linspace(tmin,tmax,npts)
#
#       sol = solve_ivp(func_ode, t_span=[tmin, tmax], y0=X0, method=method, args=args, events=event_upperbound, t_eval=t_eval)
#
#       T = sol.t
#       X = sol.y[0]
#       Y = sol.y[1]
#       Z = sol.y[2]
#       return T, X, Y, Z
#
#       # lower bound y = -1, upper bound y = D-1 (check condition on determinant for possible error).
#
# def wave_front_get_ode_sol_old(C, p0=-0.99, tmin=0, tmax=1000, npts=1000, t_eval=None):
#     from scipy.integrate import solve_ivp
#
#     def func_ode(t, x, *f_args):
#       """
#       function to integrate
#       """
#       C = f_args[0]
#       q, p = x
#
#       return np.array([-q/(1.+p) + C*(1.-p), q,])
#
#     def event_upperbound(t, x, *f_args):
#         return x[1]+p0
#
#     event_upperbound.terminal=True
#
#     q0 = C*(1-p0**2)
#     X0=np.array([q0,p0])
#     args = [C]
#
#     if t_eval is None:
#       t_eval = np.linspace(tmin,tmax,npts)
#
#     sol = solve_ivp(func_ode, t_span=[tmin,tmax], y0=X0, method='DOP853', args=args, \
#         events=event_upperbound, t_eval=t_eval)
#
#     T = sol.t
#     X = sol.y[0]
#     Y = sol.y[1]
#     return T, X, Y
#
# #==============================================================================
# # fourrier transforms
# #==============================================================================
# def get_array_module(phi):
#   """
#   Return the module of phi
#   """
#   if cp:
#     return cp.get_array_module(phi)
#   else:
#     return np
#
# def compute_freq_mesh(shape):
#     """
#     compute frequency meshes.
#     Returns a list of matrices with shape `shape`, [K1, K2, ..., Kndim], where ndim is len(shape).
#     Ki[j1,j2,j3,...,jndim] = ji/shape[i]
#     """
#     ndim = len(shape)
#
#     # store fourier frequencies ranges
#     kranges = []
#     for i in range(ndim):
#         krange = np.fft.fftfreq(shape[i])
#         kranges.append(krange)
#
#     # build meshgrid of fourier frequencies
#     Ks = [None for i in range(ndim)]
#     return np.array(np.meshgrid(*kranges, indexing='ij'))
#
# def compute_nabla_tilde(shape, a=1., reverse=False):
#     """
#     compute the nabla "vector".
#       * shape: vector giving the size in each dimension of an input field.
#       * a: lattice site size. (length unit).
#     """
#
#     ndim = len(shape)
#     newshape = [ndim] + list(shape)
#     nabla_tilde = np.zeros(newshape, dtype=np.complex_)
#
#     # build meshgrid of fourier frequencies
#     Ks = compute_freq_mesh(shape)
#
#     # fill-in nabla_tilde vector
#     for d in range(ndim):
#         nabla_tilde[d] = 2*1.j*np.exp(1.j*np.pi*Ks[d])*np.sin(np.pi*Ks[d]) / a
#         # nabla_tilde[d] = 2*np.pi*1.j*Ks[d] / a
#
#     if reverse:
#         nabla_tilde = - np.conjugate(nabla_tilde)
#
#     return nabla_tilde
#
# def compute_laplacian_tilde(shape, a=1.):
#     """
#     compute the laplacian field.
#       * shape: vector giving the size in each dimension of an input field.
#       * a: lattice site size. (length unit).
#     """
#
#     # build meshgrid of fourier frequencies
#     Ks = compute_freq_mesh(shape)
#
#     return -4.*np.sum(np.sin(np.pi*Ks)**2, axis=0)
#     # return (2.*np.pi)**2 * np.sum(Ks**2, axis=0)
#
# def compute_fft(phi, start=0):
#     """
#     Compute the fft for the input field phi.
#     Assumes that the first dimension is not to be Fourier transformed.
#     """
#     xp = get_array_module(phi)
#
#     shape = phi.shape
#     ndim = len(shape)
#
#     return xp.fft.fftn(phi, axes=range(start, ndim))
#
# def compute_ifft(phi_tilde, start=0):
#     """
#     Compute the ifft for the input field phi_tilde.
#     Assumes that the first dimension is not to be Fourier transformed.
#     Assumes that the returned field must be real-valued.
#     """
#     xp = get_array_module(phi_tilde)
#
#     shape = phi_tilde.shape
#     ndim = len(shape)
#
#     return xp.real(xp.fft.ifftn(phi_tilde, axes=range(start, ndim)))
#
# def laplacian_discrete(X):
#     """
#     Compute the discrete laplacian of the matrix X such that:
#       * there are dirichlet boundary conditions along the first axis
#       * there are periodic boundary conditions along the second axis
#     """
#     xp = cp.get_array_module(X)
#
#     # discrete laplacian with dirichlet boundary conditions
#     Dx = xp.diff(X, append=0, axis=0) - xp.diff(X, prepend=0, axis=0)
#
#     # discrete laplacian with periodic boundary conditions
#     Dy = xp.diff(X, append=X[:,0].reshape(X.shape[0],1), axis=1) - xp.diff(X, prepend=X[:,-1].reshape(X.shape[0],1), axis=1)
#
#     return Dx + Dy
#
# def laplacian_discrete_conv(X, kernel_=np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])):
#     """
#     Compute the discrete laplacian of the matrix X such that:
#       * there are dirichlet boundary conditions along the first axis
#       * there are periodic boundary conditions along the second axis
#
#       The 7-pt stencil would be [[1, 2, 1], [2,-12,2], [1,2,1]]/12
#     """
#     xp = cp.get_array_module(X)
#     kernel = xp.array(kernel_)
#
#     kp = np.array(kernel.shape) // 2
#     xshape = np.array(X.shape)
#     Y = xp.zeros(tuple(xshape + 2*kp), dtype=X.dtype)
#     Y[kp[0]:kp[0]+xshape[0], kp[1]:kp[1]+xshape[1]] = X
#
#     # boundary conditions
#     ## Dirichlet along X
#     Y[kp[0]-1] = 0.
#     for i in range(1, kp[0]):
#         Y[kp[0]-1-i] = - Y[kp[0]+i-1]
#     Y[kp[0]+xshape[0]] = 0.
#     for i in range(1, kp[0]):
#         Y[kp[0]+xshape[0]+i] = - Y[kp[0]+xshape[0]-i]
#     ## Periodic along Y
#     Y[:, :kp[1]] = Y[:, xshape[1]:kp[1]+xshape[1]]
#     Y[:, -kp[1]:] = Y[:, kp[1]:2*kp[1]]
#
#     view_shape = tuple(np.subtract(Y.shape, kernel.shape) + 1) + kernel.shape
#     strides = Y.strides + Y.strides
#
#     sub_matrices = xp.lib.stride_tricks.as_strided(Y,view_shape,strides)
#
# #     print(sub_matrices.shape, kernel.shape)
#     return xp.einsum('ij,klij->kl',kernel,sub_matrices)

