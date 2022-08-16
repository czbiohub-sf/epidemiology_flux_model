#!/bin/bash
# This script execute all notebooks. It must be run from within the docker container.

rdir=$(pwd)
cd notebooks/
python3 5-LSQ_fit_scales_exec.py
python3 6-LSQ_fit_scales_simplified_exec.py
python3 7-LSQ_fit_optimal_betamat_exec.py
python3 9-lattice_2d_nneighbors_exec.py
cd $rdir

for fname in 1-clustering \
             2-fluxes \
             3-import_cssegi \
             4-distances \
             5-LSQ_fit_scales \
             6-LSQ_fit_scales_simplified \
             7-LSQ_fit_optimal_betamat \
             8-distance_cutoff_policy \
             9-lattice_2d_nneighbors \
             10-lattice_2d_pulled_wave \
             11-ode_traveling_front \
             12-infectivity_matrix_properties
do
  echo Executing notebook ${fname}.ipynb
  jupyter nbconvert --to notebook --inplace --execute notebooks/${fname}.ipynb
done

