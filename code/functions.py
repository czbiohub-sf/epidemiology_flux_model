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
from dateutil.relativedelta import relativedelta
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

#==============================================================================
# helpers methods
#==============================================================================
def get_binned(X, Y, edges):
    nbins = len(edges)-1
    digitized = np.digitize(X,edges)
    Y_subs = [None for n in range(nbins)]
    for i in range(1, nbins+1):
        Y_subs[i-1] = np.array(Y[digitized == i])

    return Y_subs

def get_array_module(a):
  """
  Return the module of an array a
  """
  if cp:
    return cp.get_array_module(a)
  else:
    return np

def geo_dist(M1, M2):
    """
    Input in degrees.
    M = (latitude, longitude)
    """
    p1, l1 = M1
    p2, l2 = M2
    pbar = 0.5*(p1+p2)

    dp = (p1-p2)*np.pi/180.
    dl = (l1-l2)*np.pi/180.
    pbar = 0.5*(p1+p2)*np.pi/180.
    return np.sqrt(dp**2+np.cos(pbar)*dl**2)

def unfold_positive(E, n=2**5, deg=12, nbins=2**6, aper=0.):
  """
  Based on: https://www.mathworks.com/matlabcentral/fileexchange/24122-unfoldingpositive

  INPUT:
    E: matrix of size nsamples x N
    n: integer (typical value 10 to 40)
    deg: integer (typical value 7 to 15)
    nbins: integer (typical value 40 to 80)
  OUTPUT:
    X: centers of bins
    Y: histogram of nearest-neighbor differences (unfolded)

  This code unfolds a positive sequence of 'N' eigenvalues for 'nsamples'
  matrix samples through polynomial fitting of the cumulative
  distribution.
  The fitting polynomial has degree 'deg'.
  The code takes as input a matrix E of size (nsamples x N) where row j
  contains the N positive eigenvalues of the j-th sample, and a number
  'n' of points between 0 and Ymax=max(max(E)).
  The cumulative distribution is computed over the 'n' points in the
  vector YR as the fraction of eigenvalues lying below YR(j) and stored
  in the vector CumDist.
  Then a polynomial fitting is performed over the cumulative density
  profile obtained in this way, and the resulting polynomial is then
  computed on all the entries of E (---> xiMatr1).
  The nearest-neighbor difference between the unfolded eigenvalues in xiMatr1 is then computed,
  and produces a normalized histogram Y with 'nbins' (number of bins) centered at X,
  ready to be plotted.

  """
  from numpy.polynomial import Polynomial

  nsamp = E.shape[0]
  N = E.shape[1]

  ymax = np.max(np.ravel(E))
  Yr = np.linspace(0,ymax,n)

  # compute cumulative distribution
  Z = (1./float(nsamp)/float(N)) * np.array([ np.sum(np.int_(E < y)) for y in Yr], dtype=np.float_)

  # theta = np.polyfit(Yr, Z, deg)
  # p = np.poly1d(theta)
  p = Polynomial.fit(Yr, Z, deg)
  FitDist = p(Yr)
  xiMatr1 = p(E)

  d = np.ravel(np.diff(xiMatr1, axis=1))
  qlo, qhi = np.percentile(d, aper*0.5), np.percentile(d, 100.-aper*0.5)
  idx = (d >= qlo) & (d <= qhi)
  return np.histogram(d[idx] ,bins=nbins,density=True)

#==============================================================================
# modelling
#==============================================================================
########## fitting methods ##########
def fsigmoid(x, *params):
    a, b, c, d = list(params)
    return a / (1.0 + np.exp(d*(x-b))) + c

def fsigmoid_jac(x, *params):
    a, b, c, d = list(params)
    grad = np.zeros((len(params), len(x)))
    grad[0] = 1./(1.0 + np.exp(d*(x-b)))
    grad[1] = d*np.exp(d*(x-b)) * a /(1.0 + np.exp(d*(x-b)))**2
    grad[2] = np.ones(len(x))
    grad[3] = -(x-b)*np.exp(d*(x-b)) * a /(1.0 + np.exp(d*(x-b)))**2
    return grad.T

def framp2(x, a, b, c, d):
    return c*np.logaddexp(0, a*(x-b)) + d

def framp2_jac(x, a, b, c, d):
    g = np.exp(-a*(x-b))
    J1 = (x-b)*c/(1.+g)
    J2 = -a*c/(1.+g)
    J3 = np.logaddexp(0, a*(x-b))
    J4 = np.ones(len(x))
    return np.array([J1, J2, J3, J4]).T

# ########## Utils ##########
def read_df(t, tfmt, store, path):
  """
  Read a matrix present in a `store` at a certain `path` with
  the appropriate formatting of the date `t`.
  """
  key = Path(path) / t.strftime(tfmt)
  df = store[str(key)]
  return df

def get_infectivity_matrix(F, vscales=None):
  """
  Return the infectivity matrix from the input flux matrix
  """
  N = F.shape[0]
  if (F.shape[1] != N):
    raise ValueError

  if vscales is None:
    vscales = np.ones(N, dtype=np.float_)

  pvec = F.diagonal()  # populations
  pinv = np.zeros(N, dtype=np.float_)
  idx = pvec > 0.
  pinv[idx] = 1./pvec[idx]

  L = np.zeros((N,N), dtype=np.float_)
  # L = F + F.T
  FS = np.einsum('ab,b->ab', F, vscales)
  L = FS + FS.T

  np.fill_diagonal(L, pvec*vscales)
  L = np.einsum('ab,a->ab', L, pinv)

  # # symmetrize it
  # L = 0.5*(L+L.T)

  vmax = np.max(L.diagonal())
  return L/vmax

# ########## SIR integration ##########
def sir_X_to_SI(X, N):
  SI = X.reshape((2,N))
  return SI[0],SI[1]

def sir_X_to_SI_lattice_2d(X, n1, n2):
  S, I = X.reshape(2, 2**n1, 2**n2)
  return S, I

def sir_SI_to_X(S,I):
  xp = cp.get_array_module(S)
  return xp.ravel(xp.array([S,I]))

def func_sir_dX(t, X, B, g):
  """
  X: S, I
  B: localization matrix
  g: inverse recovery time
  """
  N = B.shape[0]
  S,I = sir_X_to_SI(X, N)

  dS = -np.einsum('i,ij,j->i', S, B, I)
  dI = -dS - g*I

  return sir_SI_to_X(dS,dI)

def func_sir_dV(t, X, B, g, s0):
  Y = 1. - s0*np.exp(-X)
  return np.einsum('ab,b', B, Y) - g*X

def guess_scale(V0, V1_real, B, g, Si_real, dt=1.):
  """
  Return a guess for the scale parameter
  """
  T0 = 1. - Si_real*np.exp(-V0)
  BT = np.einsum('ab,b', B, T0)
  A = V1_real - (1. - g*dt)*V0

  NUM = np.einsum('a,a', BT, A)
  DENUM = np.einsum('a,a', BT, BT)
  return NUM / DENUM / dt

def jac(X, B, g):
  N = B.shape[0]
  SI = X.reshape((2,N))
  S = SI[0]
  I = SI[1]

  # derivative of f_S
  A1 = -  np.diag(np.einsum('ij,j->i', B, I))
  A2 = - np.einsum('ij,i->ij', B, S)
  A = np.concatenate([A1, A2], axis=1)

  # derivative of f_I
  B1 = -A1
  B2 = -A2 - g*np.eye(N)
  B = np.concatenate([B1, B2], axis=1)

  return np.concatenate([A,B], axis=0)

def get_sir_omega_X(X, P):
  """
  Compute the total fraction of T=I+R individuals from local fractions and local populations
  """
  N = len(P)
  return np.einsum('i,i', 1.-X.reshape(2, N)[0], P)/np.einsum('i->', P)

def get_sir_omega_SI(S, I, P):
  """
  Compute the total fraction of I+R individuals from local fractions and local populations
  """
  return get_sir_omega_X(sir_SI_to_X(S, I), P)

def get_dTs(Ss, population):

  dTs = np.diff(1.-Ss, axis=0)
  dTs = np.concatenate([1.-Ss[0].reshape(1,-1), dTs], axis=0)
  dT_tot = np.einsum('ta,a->t', dTs, population) / np.sum(population)
  return dTs, dT_tot

def integrate_sir(Xi, times, scales, gamma, store, pathtoloc, tfmt='%Y-%m-%d', method_solver='DOP853', verbose=True):
  """
  Integrate the dynamics of the SIR starting from
  the initial condition (`Xi`, `times[0]`).
  The method assumes that in the `store` at the indicated `path`, there are entries
  in the format %Y-%m-%d that described the infectivity matrices
  for the times `times[:-1]`. The array `scales` contains the scales to apply to each infectivity matrix.

  OUTPUT:
    * Xs
    * ts

  For the output the dumping interval is 1 day.
  """
  # initializations
  nt = len(times)
  t = times[0]
  X = Xi[:]
  B = read_df(t, tfmt, store, pathtoloc).to_numpy()
  N = B.shape[0]

  if len(scales) != nt - 1:
    raise ValueError("`scales` must be of length {:d}".format(nt-1))

  ts = [t]
  Xs = [X]

  for i in range(1, nt):
    if verbose:
      print(f'Integrating day {t}')
    mykey = Path(pathtoloc) / t.strftime(tfmt)
    mykey = str(mykey)
    if mykey in store.keys():
      B = read_df(t, tfmt, store, pathtoloc).to_numpy()
    elif verbose:
      print("Infectivity matrix not updated!")
    tnew = times[i]
    dt = int((tnew - t).days)
    t_range = np.arange(dt+1)
    sol = scipy.integrate.solve_ivp(func_sir_dX, y0=X, t_span=(0,dt), \
                                    t_eval=t_range, vectorized=True, args=(B*scales[i-1], gamma), \
                                    method=method_solver)

    # break conditions
    if not (sol.success):
      raise ValueError("integration failed!")

    Xnew = sol.y[:,-1]

    # dump
    Xs += [x for x in sol.y[:, 1:].T]
    ts += [t + datetime.timedelta(days=int(x)) for x in t_range[1:]]

    # update
    t = tnew
    X = Xnew

  if verbose:
    print("Integration complete")

  SIs = np.array([sir_X_to_SI(x, N) for x in Xs])
  Ss = SIs[:,0]
  Is = SIs[:,1]

  return ts, Ss, Is

########## wave analysis methods ##########
def wave_ode(t, X, *f_args):
  """
  Right hand side of the wave equation ODE
  """
  gamma = f_args[0]
  vc = 2.*np.sqrt(1-gamma)
  f, h, g = X

  return np.array([-(vc/g)*f  + (gamma/g - 1.)*h, f, -f + gamma/vc * h])

def wave_front_get_ode_sol(X0, gamma, zmax=1000, npts=1000, t_eval=None, s0=1., eps=1.0e-10, method='BDF'):
    from scipy.integrate import solve_ivp

    vc = 2 * np.sqrt(1. - gamma)
    def event_upperbound(t, x, *f_args):
        return (-x[0] + gamma / vc *x[1]) - eps  # check derivative of S
        # return x[2]/ftol - s0
        # return (x[0]*vc/x[2] + (gamma/x[2] - 1.)*x[1]) - 1.0e-10
        # return x[0] + 1.0e-3

    if eps < 0.:
      event_upperbound.terminal=False
    else:
      event_upperbound.terminal=True

    func_ode = wave_ode
    args = [gamma]

    if t_eval is None:
      t_eval = np.linspace(0,zmax,npts)

    sol = solve_ivp(func_ode, t_span=[t_eval[0],t_eval[-1]], y0=X0, method=method, args=args, \
        events=event_upperbound, t_eval=t_eval)
    # sol = solve_ivp(func_ode, t_span=[t_eval[0],t_eval[-1]], y0=X0, method=method, args=args, \
    #       t_eval=t_eval)

    Z = sol.t
    I_der = sol.y[0]
    I = sol.y[1]
    S = sol.y[2]

    S_der = (-I_der + gamma/vc*I)
    return Z, S, I, S_der

########## lattice simulation methods ##########
def laplacian_discrete(X, xlo=0., xhi=0.):
    """
    Compute the discrete laplacian of the matrix X such that:
      * there are dirichlet boundary conditions along the first axis
      * there are periodic boundary conditions along the second axis
    """
    xp = cp.get_array_module(X)

    # discrete laplacian with dirichlet boundary conditions
    Dx = xp.diff(X, append=xhi, axis=0) - xp.diff(X, prepend=xlo, axis=0)

    # discrete laplacian with periodic boundary conditions
    Dy = xp.diff(X, append=X[:,0].reshape(X.shape[0],1), axis=1) - xp.diff(X, prepend=X[:,-1].reshape(X.shape[0],1), axis=1)

    return Dx + Dy

def laplacian_discrete_conv(X, kernel_=np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]]), xlo=0., xhi=0.):
    """
    Compute the discrete laplacian of the matrix X such that:
      * there are dirichlet boundary conditions along the first axis
      * there are periodic boundary conditions along the second axis

    The 9-pt stencil would be [[1, 2, 1], [2,-12,2], [1,2,1]]/4
    Give same result as `laplacian_discrete` when using the 5-pt stencil.
    """
    xp = cp.get_array_module(X)
    kernel = xp.array(kernel_)

    kp = np.array(kernel.shape) // 2
    xshape = np.array(X.shape)
    Y = xp.zeros(tuple(xshape + 2*kp), dtype=X.dtype)
    Y[kp[0]:kp[0]+xshape[0], kp[1]:kp[1]+xshape[1]] = X

    # boundary conditions
    ## Dirichlet along X
    Y[:kp[0]] = xlo
    Y[xshape[0]+kp[0]:] = xhi
    # ## Dirichlet along X
    # Y[kp[0]-1] = 0.
    # for i in range(1, kp[0]):
    #     Y[kp[0]-1-i] = - Y[kp[0]+i-1]
    # Y[kp[0]+xshape[0]] = 0.
    # for i in range(1, kp[0]):
    #     Y[kp[0]+xshape[0]+i] = - Y[kp[0]+xshape[0]-i]
    ## Periodic along Y
    Y[:, :kp[1]] = Y[:, xshape[1]:kp[1]+xshape[1]]
    Y[:, -kp[1]:] = Y[:, kp[1]:2*kp[1]]

    view_shape = tuple(np.subtract(Y.shape, kernel.shape) + 1) + kernel.shape
    strides = Y.strides + Y.strides

    sub_matrices = xp.lib.stride_tricks.as_strided(Y,view_shape,strides)

#     print(sub_matrices.shape, kernel.shape)
    return xp.einsum('ij,klij->kl',kernel,sub_matrices)

def get_residual_susceptible(gamma_tilde, s0=1.):
  """
  Return the residual fraction of susceptible individuals.
  This is the renormalized SIR equation with beta = 1.
  """
  if gamma_tilde < 0.:
      raise ValueError("gamma > 0!")
  elif (gamma_tilde == 0.):
      return 0.
  else:
    from scipy.optimize import root_scalar

    func_root = lambda x: s0*np.exp(-x) + gamma_tilde*x - 1.
    xlo = np.log(s0/gamma_tilde)
    xhi = 1./gamma_tilde
    sol = root_scalar(func_root, bracket=[xlo, xhi], method='brentq')
    xinf = sol.root
    sinf = s0*np.exp(-xinf)
    return sinf

def lattice_2d_ode(t, X, gamma, N1, N2, Ilo=0., Ihi=0.):
    """
    function to integrate for the nearest neighbor SIR dynamics on a 2d lattice.
    """
    # extract S and I
    S, I = cp.reshape(cp.array(X), (2, N1, N2), order='C')

    kernel = np.array([[1,2,1],[2,-12,2],[1,2,1]], dtype=np.float_)/4.  #9-pt stencil for the Laplacian computation
    # kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float_)

    # compute U
    lap_I = laplacian_discrete_conv(I, kernel, xlo=Ilo, xhi=Ihi)
    # lap_I = laplacian_discrete(I, xlo=Ilo, xhi=Ihi)
    U = lap_I + I

    dS = -S*U
    dI = +S*U - gamma*I

    dX = cp.ravel(cp.array([dS, dI]), order='C')
    return dX.get()

def lattice_2d_event_upperbound(t, X, gamma, N1, N2, Ilo=0., Ihi=0.):
    eps = 1.0e-5
    S, I = np.reshape(X, (2, N1, N2), order='C')
    return eps - np.mean(I[-1])

def lattice_2d_plot_profile(times, X_list, idump, ylabel, mass, \
        figsize=(8,3), cmap=cm.rainbow, fileout=None, dpi=300):

    Nt, N = X_list.shape

    ## make color mapping
    norm = mco.Normalize(vmin=times[0], vmax=times[-1])

    ## make figure
    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.gca()

    ### plot profiles
    for k in np.arange(Nt)[::idump]:
        t = times[k]
        Xm = X_list[k]
        color = cmap(norm(t))
        ax.plot(np.arange(N), Xm, 'o-', color=color, lw=0.5, ms=1)
        i = np.argmin(np.abs(Xm - mass))
        # print(np.min(Xm), np.max(Xm), mass, i, Xm[i])
        ax.plot(i, [Xm[i]], 'o', color=color, ms=4)

    ## axes properties
    ax.set_xlim(0, None)
    ax.set_ylim(0., None)
    ax.set_ylabel(ylabel, fontsize='medium')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(length=4)
    fig.tight_layout(rect=[0.,0.,0.98,1.])

    cax = fig.add_axes(rect=[0.98,0.2,0.01,0.7])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=cax, extendfrac='auto')
    cbar.set_label("time")


    if fileout is None:
        plt.show()
    else:
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(fileout)
        fig.clf()
        plt.close('all')

def lattice_2d_plot_position(times_list, Xs_list, mass, \
        ftol = 0.95, figsize=(4,3), fileout=None, dpi=300):
    from scipy.interpolate import UnivariateSpline
    from numpy.polynomial import Polynomial

    n = len(times_list)
    n_list = np.arange(n)
    norm = mco.Normalize(vmin=n_list[0], vmax=n_list[-1])
    cmap = cm.viridis
    colors = cmap(norm(n_list))

    ## make figure
    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.gca()

    for n in n_list:
        times = times_list[n]
        Xs = Xs_list[n]
        color=colors[n]
        Nt, N = Xs.shape

        # position
        Z = np.argmin((Xs - mass[n])**2, axis=1)

        ax.plot(times, Z, '-', color=color, lw=0.5)

        spl = UnivariateSpline(times, Z)
        # ax.plot(times, spl(times), '--', color='red', lw=0.5)
        # ax.plot(times, spl.derivative()(times), '--', color='red', lw=0.5)
        # ax.plot(times, spl.derivative().derivative()(times), '--', color='red', lw=0.5)
        # i0 = np.argmax(spl.derivative().derivative()(times))
        # ax.axvline(x=times[i0], color='k', ls='-', lw=1.)
        # i0 = np.argwhere((spl.derivative()(times)/np.max(spl.derivative()(times)) > ftol)).ravel()[0]
        i0 = np.argwhere((spl.derivative()(times)/spl.derivative()(times[-1]) > ftol)).ravel()[0]
        # ax.axvline(x=times[i0], color='k', ls='-', lw=1.)

        # c = Polynomial.fit(times[i0:], Z[i0:], deg=1)
        a, b = np.polyfit(times[i0:], Z[i0:], deg=1)
        xdata = np.array([times[i0], times[-1]])
        ax.plot(xdata, a*xdata+b, '--', lw=1, color=color, label="v = {:.2f}".format(a))

    ## axes properties
    ax.legend(loc='best', fontsize='medium')
    ax.set_xlim(0, None)
    ax.set_ylim(0., None)
    ax.set_ylabel('site', fontsize='medium')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(length=4)
    fig.tight_layout()

    if fileout is None:
        plt.show()
    else:
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(fileout)
        fig.clf()
        plt.close('all')

def lattice_2d_plot_velocities(gamma_list, times_list, Xs_list, mass, \
        ftol = 0.95, figsize=(4,3), fileout=None, dpi=300):
    from scipy.interpolate import UnivariateSpline
    from numpy.polynomial import Polynomial

    n = len(times_list)
    n_list = np.arange(n)

    velocities = []
    for n in n_list:
        times = times_list[n]
        Xs = Xs_list[n]
        Nt, N = Xs.shape

        # position
        Z = np.argmin((Xs - mass[n])**2, axis=1)

        spl = UnivariateSpline(times, Z)
        i0 = np.argwhere((spl.derivative()(times)/spl.derivative()(times[-1]) > ftol)).ravel()[0]

        a, b = np.polyfit(times[i0:], Z[i0:], deg=1)
        velocities.append(a)

    ## make figure
    fig = plt.figure(facecolor='w', figsize=figsize, dpi=dpi)
    ax = fig.gca()

    ax.plot(gamma_list, velocities, '-o', color='darkblue', lw=0.5, ms=2)

    Xfit = np.linspace(0., 1., 1000)
    Yfit = 2.*np.sqrt(1.-Xfit)
    ax.plot(Xfit, Yfit, '--', color='red', lw=0.5)

    ## axes properties
    # ax.legend(loc='best', fontsize='medium')
    ax.set_xlim(0, None)
    ax.set_ylim(0., None)
    ax.set_xlabel('gamma', fontsize='medium')
    ax.set_ylabel('velocity', fontsize='medium')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(length=4)
    fig.tight_layout()

    if fileout is None:
        plt.show()
    else:
        fig.savefig(fileout, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(fileout)
        fig.clf()
        plt.close('all')

#==============================================================================
# plot methods
#==============================================================================
def show_image(mat_, downscale=None, log=False, mpl=True, vmin=None, vmax=None, fileout=None, dpi=72, interpolation='none', method='sum'):
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

def plot_omega_profile(Omegas, times, labels=None, colors=None, styles=None, fileout=Path('./animation.gif'), tpdir=Path('.'), \
                       dpi=150, lw=0.5, ms=2, \
                       ymin=None, ymax=None, figsize=(4,3), fps=5, duration=10, \
                       log=True, xlabel='community', ylabel="$\Omega_a$", lgd_ncol=2, deletetp=True, exts=['.png'], \
                       tfmt = "%Y-%m-%d", verbose=False):
  """
  Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
  representing the dynamics of local epidemic sizes
  See this tutorial on how to make animated movies:
    https://matplotlib.org/stable/api/animation_api.html
  INPUT:
    * Omegas: list of table containing omegas (indices t,a)
    * times: list of times (indices t)
  """
  # tp dir
  if not tpdir.is_dir():
    tpdir.mkdir(exist_ok=True)
  for ext in exts:
    for f in tpdir.glob('*' + ext): f.unlink()

  # parameters
  nseries = Omegas.shape[0]
  nt = len(times)
  if (Omegas.shape[1] != nt):
    raise ValueError("Omegas must have same second dimension as times!")
  N = Omegas.shape[2]

  haslabels=True
  if labels is None:
    haslabels=False
    labels = [None]*nseries

  if colors is None:
    colors = [None]*nseries

  if styles is None:
    styles = [None]*nseries
  for k in range(nseries):
    if styles[k] is None:
      styles[k] = 'o'

  num = int(np.ceil(np.log10(nt)))
  if float(nt) == float(10**num):
    num += 1
  # fmt = "{" + ":0{:d}".format(num) + "}"

  # determine minimum and maximum
  idx = Omegas[:,0,:] > 0.
  if ymin is None:
    ymin = 10**(np.floor(np.log10(np.min(Omegas[:,0,:][idx]))))    # closest power of 10
  if ymax is None:
    ymax = 10**(np.ceil(np.log10(np.max(Omegas))))    # closest power of 10
  if verbose:
    print("ymin = {:.2e}".format(ymin), "ymax = {:.2e}".format(ymax))

  if not ".png" in exts:
    raise ValueError("PNG format must be given")

  # determine dumping interval
  nframes = duration*fps
  idump = int(np.ceil(nt / nframes))

  # community index
  X = np.arange(N, dtype=np.uint)

  # prepare figure
  filenames=[]
  for i in np.arange(nt)[::idump]:
    ## update time and Omega
    t = times[i]

    ## create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    date = t.strftime('%Y-%m-%d')
    title = "{:s}".format(date)
    ax.set_title(title, fontsize="large")

    for k in range(nseries):
      Y = Omegas[k, i]
      color=colors[k]
      label=labels[k]
      style=styles[k]
      ax.plot(X, Y, style, lw=lw, mew=0, ms=ms, color=color, label=label)

    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel(xlabel, fontsize='medium')
    ax.set_ylabel(ylabel, fontsize='large')
    if log:
      ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(length=4)

    if haslabels:
      ax.legend(loc='lower left', fontsize='medium', frameon=False, ncol=lgd_ncol)

    fname = str(tpdir / t.strftime(tfmt))
    # fname = str(tpdir / fmt.format(i))
    for ext in exts:
      fpath = fname + ext
      fig.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    fpath = fname + ".png"
    filenames.append(fpath)

    if verbose:
      print(f"Written file {fpath}.")

    fig.clf()
    plt.close('all')

  # write movie
  imageio.mimsave(fileout, [imageio.imread(f) for f in filenames], fps=fps)
  print(f"Written file {fileout}.")

  # clean tpdir
  if deletetp:
    shutil.rmtree(tpdir)

  return

def plot_omega_map(Omega, times, XY, fileout=Path('./animation.gif'), tpdir=Path('.'), dpi=150, \
                   vmin=None, vmax=None, figsize=(4,3), nframes=None, fps=10, duration=10, \
                   cmap=cm.magma_r, tfmt = "%Y-%m-%d", ymin=None, ymax=None, \
                   clabel='$\Omega$', deletetp=True, exts=['.png'], \
                   circle_size=0.4, lw=0.1, edges=[], edge_width=0.5, verbose=False):
  """
  Save an animated image series (GIF) or movie (MP4), depending on the extension provided,
  representing the dynamics of local epidemic sizes

  INPUT:
    * df_tolls: list of dataframes
    * XY: 2xN array giving the coordinates of the N communities.
    *
  """
  from matplotlib.path import Path
  # tp dir
  if not tpdir.is_dir():
    tpdir.mkdir(exist_ok=True)
  for ext in exts:
    for f in tpdir.glob('*' + ext): f.unlink()

  # parameters
  nt = len(times)
  if (Omega.shape[0] != nt):
    raise ValueError("Omega must have same second dimension as times!")
  N = Omega.shape[1]

  num = int(np.ceil(np.log10(nt)))
  if float(nt) == float(10**num):
    num += 1
  fmt = "{" + ":0{:d}".format(num) + "}"

  # color scale
  # determine minimum and maximum
  idx = Omega[0,:] > 0.
  if vmin is None:
    vmin = 10**(np.floor(np.log10(np.min(Omega[0,:][idx]))))    # closest power of 10
  if vmax is None:
    vmax = 10**(np.ceil(np.log10(np.max(Omega))))    # closest power of 10
  if verbose:
    print("vmin = {:.2e}".format(vmin), "vmax = {:.2e}".format(vmax))
  norm = mco.LogNorm(vmin=vmin, vmax=vmax)

  # determine dumping interval
  nframes = duration*fps
  idump = int(np.ceil(nt / nframes))

  # clusters
  X, Y = XY
  xmin = np.min(X)
  xmax = np.max(X)
  if ymin is None:
    ymin = np.min(Y)
  if ymax is None:
    ymax = np.max(Y)

  # prepare figure
  filenames=[]
  for i in np.arange(nt)[::idump]:
    ## update time and Omega
    t = times[i]

    ## create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.gca()

    date = t.strftime('%Y-%m-%d')
    title = "{:s}".format(date)
    ax.set_title(title, fontsize="large")

    # draw edges
    if len(edges) > 0:
      for a1,a2 in edges:
        x1 = X[a1]
        y1 = Y[a1]
        x2 = X[a2]
        y2 = Y[a2]
        # ax.plot([x1,x2], [y1,y2], 'k-', lw=edge_width)
        verts = [ (x1, y1), (x2, y2)]
        codes = [Path.MOVETO, Path.LINETO]
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor='k', lw=edge_width)
        res = ax.add_patch(patch)

    # draw spheres
    Ns = np.arange(N)
    idx = np.argsort(Omega[i])
    # for a in range(N):
    for a in Ns[idx]:
      x = X[a]
      y = Y[a]
      val = Omega[i,a]
      if (val < vmin):
        color = [1.,1.,1.,1.]
      elif (val > vmax):
        color = [0.,0.,0.,1.]
      else:
        color = cmap(norm(val))
      circle = plt.Circle((x,y), circle_size, color=color, alpha=1, lw=lw, ec='black')
      res = ax.add_patch(circle)

    # formatting
    for lab in 'left', 'right', 'bottom', 'top':
      ax.spines[lab].set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    cax = fig.add_axes(rect=[0.98,0.1,0.02,0.7])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=clabel, extendfrac='auto')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    # write figure
    fname = str(tpdir / t.strftime(tfmt))
    for ext in exts:
      fpath = fname + ext
      fig.savefig(fpath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    fpath = fname + ".png"
    filenames.append(fpath)

    if verbose:
      print(f"Written file {fpath}.")

    fig.clf()
    plt.close('all')

  # write movie
  imageio.mimsave(fileout, [imageio.imread(f) for f in filenames], fps=fps)
  print(f"Written file {fileout}.")

  # clean tpdir
  if deletetp:
    shutil.rmtree(tpdir)

  return

def plot_scatter(times, Xs, Ys, idump=1, vmin=None,vmax=None, \
                 xlabel=None, ylabel=None, logscale=False, fpaths=['plot_scatter.png'], \
                 cmap=cm.rainbow, figsize=(4,3), dpi=300, alpha=0.3, verbose=True):

    Nt = len(times)
    if Nt == 1:
      colors = ['darkblue']
    else:
      ## color mapping with date value
      indices = np.arange(Nt)
      norm = mco.Normalize(vmin=np.min(indices), vmax=np.max(indices))
      colors = cmap(norm(indices))

    ## vmin, vmax
    if vmin is None:
        xmin = np.nanmin(Xs)
        ymin = np.nanmin(Ys)
        vmin = min(xmin, ymin)
    if vmax is None:
        xmax = np.nanmax(Xs)
        ymax = np.nanmax(Ys)
        vmax = max(xmax, ymax)

    ## make figure
    fig = plt.figure(facecolor='w', figsize=figsize, dpi=dpi)
    ax = fig.gca()

    for i in np.arange(Nt)[::idump]:
        t = times[i]
        X = Xs[i]
        Y = Ys[i]

        ax.plot(X, Y, 'o', color=colors[i], lw=0, mew=0, ms=2, alpha=alpha)

    ax.plot([vmin, vmax], [vmin, vmax], 'k-', lw=0.5)
    # plot formatting
    ax.set_xlabel(xlabel, fontsize='medium')
    ax.set_ylabel(ylabel, fontsize='medium')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(length=4)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_aspect('equal')

    fig.tight_layout(rect=[0.,0.,0.95,1.])

    if Nt > 1:
      cax = fig.add_axes(rect=[0.99,0.2,0.01,0.7])
      cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax, extendfrac='auto')
      nmonth = (times[-1].year-times[0].year)*12 + (times[-1].month-times[0].month)
      tick_values = np.array([times[0] + relativedelta(months=i) for i in range(nmonth+1)])
      tick_values = tick_values[::2]
      ticks = [times.index(t) for t in tick_values]
      # labels = [times[(t-1)*window].strftime('%Y-%m-%d') for t in np.array(ticks, dtype=np.int_)]
      labels = [t.strftime('%Y-%m-%d') for t in tick_values]
      cbar.set_ticks(ticks)
      cbar.set_ticklabels(labels)

    for fpath in fpaths:
        fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
        if verbose:
          print("Written file: {:s}".format(str(fpath)))
    fig.clf()
    plt.close('all')
    # plt.show()
    return

